"""
Hugging Face zero-shot text-classification pipeline for intent classification
- No training. Uses a pretrained NLI model (e.g., DeBERTa-v3 MNLI) to classify
  conversation-level text into four intents via zero-shot classification.
- Integrates with the IntentClassifier base.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder

from ..base.classifier import IntentClassifier
from ..base.data_structures import Conversation, Dataset as ConvDataset

logger = logging.getLogger(__name__)


class HFZeroShotPipelineClassifier(IntentClassifier):
    """
    Zero-shot classifier using HF pipeline('zero-shot-classification').
    - Candidate labels are expressed as natural language phrases.
    - No fine-tuning required; great for quick baselines and ensembles.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})

        cfg = {
            # Strong multilingual NLI model for zero-shot classification:
            # "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli-ling-wanli"
            # For a smaller/faster option: "facebook/bart-large-mnli"
            "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli-ling-wanli",
            "text_source": "full",  # 'user' | 'full'
            "hypothesis_template": "This hotel booking conversation is {}.",
            "device": 0 if torch.cuda.is_available() else -1,
        }
        cfg.update(self.config)
        self.config = cfg

        # Internal target labels (fixed set)
        self.target_labels = ["low", "medium", "high", "abandoned"]

        # Descriptive candidate labels to help zero-shot performance
        self.candidate_map = {
            "low": "low intent (exploring; broad discovery; no specific dates)",
            "medium": "medium intent (evaluating; comparing options; dates/prices)",
            "high": "high intent (actioning; ready to book; giving details)",
            "abandoned": (
                "abandoned booking (payment requested or confirmation step reached but not completed)"
            ),
        }
        self.candidate_labels = list(self.candidate_map.values())
        self.reverse_map = {v: k for k, v in self.candidate_map.items()}

        self.pipe = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_trained = False

    # ========= Base API =========

    def train(
        self, train_dataset: ConvDataset, validation_dataset: Optional[ConvDataset] = None
    ) -> Dict[str, Any]:
        """
        No training. Initializes the HF zero-shot pipeline and label encoder.
        """
        logger.info("Initializing zero-shot classification pipeline...")
        self.pipe = pipeline(
            task="zero-shot-classification",
            model=self.config["model_name"],
            device=self.config["device"],
            # tokenizer/model handle truncation internally
        )

        # Fixed encoder over all four target labels for consistent outputs
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.target_labels)

        self.is_trained = True

        # Return a compact summary for consistency with other trainers
        return {
            "training_method": "zero_shot_pipeline",
            "model_name": self.config["model_name"],
            "n_train": len(train_dataset),
        }

    def predict(self, conversations: List[Conversation]) -> List[str]:
        if not self.is_trained or self.pipe is None:
            raise ValueError("Classifier not initialized. Call train() first.")

        texts = [self._conversation_to_text(c) for c in conversations]
        results = self.pipe(
            sequences=texts,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.config["hypothesis_template"],
            multi_label=False,
        )

        # Pipeline returns a dict for single input, list of dicts for batch
        if isinstance(results, dict):
            results = [results]

        preds: List[str] = []
        for res in results:
            if not res.get("labels"):
                preds.append("low")  # safe fallback
                continue
            top_label = res["labels"][0]  # highest score
            internal = self.reverse_map.get(top_label, "low")
            preds.append(internal)

        return preds

    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        if not self.is_trained or self.pipe is None:
            raise ValueError("Classifier not initialized. Call train() first.")

        texts = [self._conversation_to_text(c) for c in conversations]
        results = self.pipe(
            sequences=texts,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.config["hypothesis_template"],
            multi_label=False,
        )

        if isinstance(results, dict):
            results = [results]

        proba_list: List[Dict[str, float]] = []
        for res in results:
            # Build zero dict then fill with returned scores
            probs = {label: 0.0 for label in self.target_labels}
            labels = res.get("labels", [])
            scores = res.get("scores", [])
            for lbl, score in zip(labels, scores):
                internal = self.reverse_map.get(lbl)
                if internal:
                    probs[internal] = float(score)

            # Normalize safety (should already be ~1.0 for multi_label=False)
            total = sum(probs.values()) or 1.0
            for k in probs:
                probs[k] = float(probs[k] / total)

            proba_list.append(probs)

        return proba_list

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