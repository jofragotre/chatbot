"""
Hugging Face Transformer-based intent classifier (sequence classification)
- Trains a transformer encoder (e.g., distilbert-base-uncased) on conversation-level text
- Integrates with the IntentClassifier base interface
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..base.classifier import IntentClassifier
from ..base.data_structures import Conversation, Dataset as ConvDataset

logger = logging.getLogger(__name__)


class _HFIntentTorchDataset(TorchDataset):
    """Torch dataset for intent classification"""

    def __init__(self, encodings: Dict[str, List[List[int]]], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class HFSequenceClassifier(IntentClassifier):
    """
    Conversation-level classifier using Hugging Face transformers.
    - Fine-tunes a sequence classification model on your four intents
    - Uses conversation text (user-only or full) as input
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})

        # Defaults
        cfg = {
            "model_name": "distilbert-base-uncased",
            "text_source": "user",  # 'user' | 'full'
            "max_length": 512,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "logging_steps": 50,
            "seed": 42,
            "output_dir": "artifacts/hf_intent_cls",
            "eval_on_validation": True,
            "early_stopping_patience": 2,  # set None/0 to disable
            "fp16": True,  # auto-disabled if CUDA unavailable
        }
        cfg.update(self.config)
        self.config = cfg

        # Components
        self.tokenizer = None
        self.model = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.trainer: Optional[Trainer] = None

        self.is_trained = False

    # ========= Base API =========

    def train(
        self, train_dataset: ConvDataset, validation_dataset: Optional[ConvDataset] = None
    ) -> Dict[str, Any]:
        """Fine-tune the HF classifier"""
        logger.info("Initializing tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Label encoding
        y_train_str = train_dataset.get_labels()
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train_str)
        else:
            y_train = self.label_encoder.transform(y_train_str)

        labels = list(self.label_encoder.classes_)
        id2label = {i: lab for i, lab in enumerate(labels)}
        label2id = {lab: i for i, lab in enumerate(labels)}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        # Build datasets
        train_texts = [self._conversation_to_text(c) for c in train_dataset.conversations]
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            max_length=self.config["max_length"],
            padding=False,
        )
        torch_train_ds = _HFIntentTorchDataset(train_encodings, labels=list(y_train))

        torch_val_ds = None
        eval_strategy = "no"
        save_strategy = "no"
        callbacks = []

        if validation_dataset is not None and len(validation_dataset) > 0 and self.config["eval_on_validation"]:
            y_val = self.label_encoder.transform(validation_dataset.get_labels())
            val_texts = [self._conversation_to_text(c) for c in validation_dataset.conversations]
            val_encodings = self.tokenizer(
                val_texts, truncation=True, max_length=self.config["max_length"], padding=False
            )
            torch_val_ds = _HFIntentTorchDataset(val_encodings, labels=list(y_val))
            eval_strategy = "epoch"
            save_strategy = "epoch"
            if self.config.get("early_stopping_patience", 0):
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config["early_stopping_patience"]))

        fp16 = bool(self.config["fp16"] and torch.cuda.is_available())

        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_train_epochs"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_ratio=self.config["warmup_ratio"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            evaluation_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=(eval_strategy != "no"),
            metric_for_best_model="eval_weighted_f1",
            greater_is_better=True,
            logging_steps=self.config["logging_steps"],
            report_to="none",
            seed=self.config["seed"],
            fp16=fp16,
        )

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8 if fp16 else None)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=torch_train_ds,
            eval_dataset=torch_val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        train_output = self.trainer.train()
        self.is_trained = True

        metrics = {"n_train": len(train_dataset)}
        # Quick train accuracy
        train_preds = self.predict(train_dataset.conversations)
        train_acc = (np.array(train_preds) == np.array(train_dataset.get_labels())).mean()
        metrics["train_accuracy"] = float(train_acc)

        if torch_val_ds is not None:
            eval_metrics = self.trainer.evaluate()
            # keep a compact subset
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[f"val_{k.replace('eval_', '')}"] = float(v)

        return metrics

    def predict(self, conversations: List[Conversation]) -> List[str]:
        """Predict labels for conversations"""
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise ValueError("Classifier not trained yet")

        texts = [self._conversation_to_text(c) for c in conversations]
        enc = self.tokenizer(texts, truncation=True, max_length=self.config["max_length"], padding=False)
        ds = _HFIntentTorchDataset(enc, labels=[0] * len(texts))  # dummy labels

        preds = self.trainer.predict(ds)
        logits = preds.predictions
        y_idx = logits.argmax(axis=1)
        return self.label_encoder.inverse_transform(y_idx)

    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        """Predict probabilities for each intent"""
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise ValueError("Classifier not trained yet")

        texts = [self._conversation_to_text(c) for c in conversations]
        enc = self.tokenizer(texts, truncation=True, max_length=self.config["max_length"], padding=False)
        ds = _HFIntentTorchDataset(enc, labels=[0] * len(texts))  # dummy labels

        preds = self.trainer.predict(ds)
        logits = preds.predictions
        probs = self._softmax(logits, axis=1)
        labels = list(self.label_encoder.classes_)

        out: List[Dict[str, float]] = []
        for row in probs:
            out.append({labels[i]: float(row[i]) for i in range(len(labels))})
        return out

    # ========= Internals =========

    def _conversation_to_text(self, conv: Conversation) -> str:
        src = self.config["text_source"]
        if src == "user":
            return conv.get_user_text()
        elif src == "full":
            parts = []
            for m in conv.messages:
                prefix = "USER:" if m.role == "user" else "BOT:"
                parts.append(f"{prefix} {m.text}")
            return " ".join(parts)
        return conv.get_user_text()

    def _compute_metrics(self, eval_pred):
        """Compute eval metrics for Trainer"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=1)

        accuracy = accuracy_score(labels, preds)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1": float(f1_weighted),
        }

    @staticmethod
    def _softmax(x, axis=1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)