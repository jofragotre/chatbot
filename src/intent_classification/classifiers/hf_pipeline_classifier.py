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

from base.classifier import IntentClassifier
from base.data_structures import Conversation, Dataset as ConvDataset

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
            "model_name": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "text_source": "full",  # 'user' | 'full'
            "hypothesis_template": "The user in this conversation is {}.",
            "device": 0 if torch.cuda.is_available() else -1,
        }
        cfg.update(self.config)
        self.config = cfg

        # Internal target labels (fixed set)
        self.target_labels = ["low", "medium", "high", "abandoned"]

        # Descriptive candidate labels to help zero-shot performance
        self.candidate_map = {
            "low": "browsing hotel information without booking commitment",
            "medium": "actively considering booking with specific requirements", 
            "high": "ready to complete a booking transaction",
            "abandoned": "started booking process but did not complete payment"
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

        self.is_trained = True

        # Return a compact summary for consistency with other trainers
        return {
            "training_method": "zero_shot_pipeline",
            "model_name": self.config["model_name"],
            "n_train": len(train_dataset),
        }

    def predict(self, conversations: List[Conversation]) -> List[str]:
        """Classify each user message then aggregate to conversation level"""
        if not self.is_trained or self.pipe is None:
            raise ValueError("Classifier not initialized. Call train() first.")
        
        results = []
        
        for conv in conversations:
            user_messages = conv.get_user_messages()
            if not user_messages:
                results.append("low")
                continue
                
            # Classify each user message
            message_texts = [msg.text for msg in user_messages]
            message_results = self.pipe(
                sequences=message_texts,
                candidate_labels=self.candidate_labels,
                hypothesis_template=self.config["hypothesis_template"],
                multi_label=False,
            )
            
            # Convert to consistent format
            if isinstance(message_results, dict):
                message_results = [message_results]
                
            # Aggregate predictions using weighted voting
            intent_scores = {label: 0.0 for label in self.target_labels}
            
            for i, msg_result in enumerate(message_results):
                # Weight later messages more heavily (recency bias)
                weight = (i + 1) / len(message_results)
                
                for label, score in zip(msg_result["labels"], msg_result["scores"]):
                    internal_label = self.reverse_map.get(label)
                    if internal_label:
                        intent_scores[internal_label] += weight * score
            
            # Return highest scoring intent
            final_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            results.append(final_intent)
        
        return results

    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        """Get probabilities using per-message classification and aggregation"""
        if not self.is_trained or self.pipe is None:
            raise ValueError("Classifier not initialized. Call train() first.")
    
        proba_list = []
        
        for conv in conversations:
            user_messages = conv.get_user_messages()
            if not user_messages:
                # Default probabilities for conversations with no user messages
                default_probs = {label: 0.0 for label in self.target_labels}
                default_probs["low"] = 1.0
                proba_list.append(default_probs)
                continue
                
            # Classify each user message
            message_texts = [msg.text for msg in user_messages]
            message_results = self.pipe(
                sequences=message_texts,
                candidate_labels=self.candidate_labels,
                hypothesis_template=self.config["hypothesis_template"],
                multi_label=False,
            )
            
            if isinstance(message_results, dict):
                message_results = [message_results]
                
            # Aggregate probabilities using weighted voting
            intent_scores = {label: 0.0 for label in self.target_labels}
            
            for i, msg_result in enumerate(message_results):
                # Weight later messages more heavily (recency bias)
                weight = (i + 1) / len(message_results)
                
                for label, score in zip(msg_result["labels"], msg_result["scores"]):
                    internal_label = self.reverse_map.get(label)
                    if internal_label:
                        intent_scores[internal_label] += weight * score
            
            # Normalize to probabilities
            total = sum(intent_scores.values()) or 1.0
            normalized_probs = {k: v / total for k, v in intent_scores.items()}
            proba_list.append(normalized_probs)
        
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