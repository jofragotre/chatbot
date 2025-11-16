"""
Classical ML + TF-IDF intent classifier
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from scipy.special import softmax

from base.classifier import IntentClassifier
from base.data_structures import Conversation, Dataset

logger = logging.getLogger(__name__)


class TfidfMLClassifier(IntentClassifier):
    """
    Conversation-level classifier using TF-IDF + Classical ML.
    - Text source configurable (user-only or full conversation)
    - Optional structural features combined with TF-IDF
    - Supports Logistic Regression or Linear SVM (+ calibration)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})

        cfg = {
            "text_source": "full",  # 'user' | 'full'
            "ngram_range": (1, 2),
            "max_features": 8000,
            "lowercase": True,
            "stop_words": "english",
            "min_df": 2,
            "max_df": 0.95,
            "use_structural_features": True,
            "model_type": "logreg",  # 'logreg' | 'svm'
            "C": 2.0,
            "class_weight": "balanced",
            "calibrate": True,  # for SVM
            "random_state": 42,
        }
        cfg.update(self.config)
        self.config = cfg

        # Components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.clf = None

        # Keep for base save/load compatibility
        self.model = None
        self.feature_extractor = None  # will store vectorizer here for pickling

        self.is_trained = False

    def train(
        self, train_dataset: Dataset, validation_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        texts, X_struct = self._prepare_inputs(train_dataset.conversations, fit=True)
        y = self._get_labels(train_dataset.get_labels(), fit=True)

        X = self._combine_features(texts, X_struct)

        self.clf = self._build_model()
        logger.info(
            f"Training TF-IDF ML classifier: model={self.config['model_type']}, "
            f"features=TFIDF{' + structural' if self.config['use_structural_features'] else ''}"
        )
        self.clf.fit(X, y)

        self.model = self.clf
        self.feature_extractor = self.vectorizer  # for saving
        self.is_trained = True

        # Basic train metrics
        train_preds = self.label_encoder.inverse_transform(self.clf.predict(X))
        train_acc = (np.array(train_preds) == np.array(train_dataset.get_labels())).mean()

        metrics = {"train_accuracy": float(train_acc), "n_train": len(train_dataset)}
        if validation_dataset is not None and len(validation_dataset) > 0:
            val_metrics = self._quick_eval(validation_dataset)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        return metrics

    def predict(self, conversations: List[Conversation]) -> List[str]:
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")

        texts, X_struct = self._prepare_inputs(conversations, fit=False)
        X = self._combine_features(texts, X_struct)
        y_pred = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")

        texts, X_struct = self._prepare_inputs(conversations, fit=False)
        X = self._combine_features(texts, X_struct)

        labels = list(self.label_encoder.classes_)
        proba_list: List[Dict[str, float]] = []

        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X)
        elif hasattr(self.clf, "decision_function"):
            scores = self.clf.decision_function(X)
            if scores.ndim == 1:  # binary case
                scores = np.vstack([-scores, scores]).T
            probs = softmax(scores, axis=1)
        else:
            # Fallback: one-hot on predictions
            pred = self.clf.predict(X)
            probs = np.zeros((len(pred), len(labels)))
            for i, p in enumerate(pred):
                probs[i, p] = 1.0

        for row in probs:
            proba = {labels[i]: float(row[i]) for i in range(len(labels))}
            proba_list.append(proba)

        return proba_list

    # --------- internals ---------

    def _prepare_inputs(
        self, conversations: List[Conversation], fit: bool
    ) -> Tuple[Any, Optional[np.ndarray]]:
        # Prepare texts
        texts = [self._conversation_to_text(c) for c in conversations]

        # TF-IDF
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.config["ngram_range"],
                max_features=self.config["max_features"],
                lowercase=self.config["lowercase"],
                stop_words=self.config["stop_words"],
                min_df=self.config["min_df"],
                max_df=self.config["max_df"],
            )
            X_text = self.vectorizer.fit_transform(texts)
        else:
            X_text = self.vectorizer.transform(texts)

        # Structural features
        X_struct = None
        if self.config["use_structural_features"]:
            X_struct = np.vstack([self._struct_features(c) for c in conversations])
        return X_text, X_struct

    def _combine_features(self, X_text, X_struct) -> Any:
        if X_struct is None:
            return X_text
        return hstack([X_text, csr_matrix(X_struct)], format="csr")

    def _get_labels(self, labels: List[str], fit: bool) -> np.ndarray:
        if fit or self.label_encoder is None:
            self.label_encoder = LabelEncoder() # Categorical to numerical labels
            y = self.label_encoder.fit_transform(labels)
        else:
            y = self.label_encoder.transform(labels)
        return y

    def _build_model(self):
        model_type = self.config["model_type"]
        C = self.config["C"]
        class_weight = self.config["class_weight"]
        random_state = self.config["random_state"]

        if model_type == "logreg":
            return LogisticRegression(
                C=C,
                max_iter=2000,
                n_jobs=None,
                solver="saga",
                multi_class="auto",
                class_weight=class_weight,
                random_state=random_state,
            )
        elif model_type == "svm":
            base = LinearSVC(C=C, class_weight=class_weight, random_state=random_state)
            if self.config.get("calibrate", True):
                # Calibrate to get probabilities
                return CalibratedClassifierCV(estimator=base, cv=3)
            return base
        else:
            raise ValueError("Unsupported model_type. Use 'logreg' or 'svm'.")

    def _conversation_to_text(self, conv: Conversation) -> str:
        src = self.config["text_source"]
        if src == "user":
            return conv.get_user_text()
        elif src == "full":
            # tag roles to preserve minimal structure
            parts = []
            for m in conv.messages:
                prefix = "USER:" if m.role == "user" else "BOT:"
                parts.append(f"{prefix} {m.text}")
            return " ".join(parts)
        else:
            return conv.get_user_text()

    def _struct_features(self, conv: Conversation) -> np.ndarray:
        user_msgs = conv.get_user_messages()
        bot_msgs = conv.get_bot_messages()

        total_msgs = conv.get_message_count()
        turns = conv.get_turn_count()
        total_words = conv.get_conversation_length()

        user_text = " ".join([m.text for m in user_msgs]).lower()

        # simple counts / ratios
        q_marks = user_text.count("?")
        ex_marks = user_text.count("!")
        digits = sum(ch.isdigit() for ch in user_text)

        # rough date/price cues
        has_month = any(
            m in user_text
            for m in [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
        )
        euro = user_text.count("€") + user_text.count("eur") + user_text.count("euro")
        price_words = sum(w in user_text for w in ["price", "rate", "cost", "fee"])

        user_ratio = (len(user_msgs) / total_msgs) if total_msgs else 0.0
        avg_user_len = (
            np.mean([m.get_word_count() for m in user_msgs]) if user_msgs else 0.0
        )

        feats = [
            total_msgs,
            turns,
            len(user_msgs),
            len(bot_msgs),
            total_words,
            user_ratio,
            avg_user_len,
            q_marks,
            ex_marks,
            digits,
            float(has_month),
            euro,
            price_words,
        ]
        return np.array(feats, dtype=float)

    def _quick_eval(self, dataset: Dataset) -> Dict[str, float]:
        preds = self.predict(dataset.conversations)
        acc = (np.array(preds) == np.array(dataset.get_labels())).mean()
        return {"accuracy": float(acc)}

    @classmethod
    def load(cls, path):
        # Ensure vectorizer is restored from feature_extractor
        instance: "TfidfMLClassifier" = super(TfidfMLClassifier, cls).load(path)
        instance.vectorizer = instance.feature_extractor
        instance.clf = instance.model
        return instance