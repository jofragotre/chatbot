"""
Core data structures for intent classification
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: str  # "user" or "bot"
    text: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def is_user_message(self) -> bool:
        """Check if message is from user"""
        return self.role == "user"
    
    def is_bot_message(self) -> bool:
        """Check if message is from bot"""
        return self.role == "bot"
    
    def get_word_count(self) -> int:
        """Get word count of message"""
        return len(self.text.split())

@dataclass
class Conversation:
    """Represents a complete conversation with intent label"""
    session_id: str
    messages: List[Message]
    intent_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert dict messages to Message objects if needed
        if self.messages and isinstance(self.messages[0], dict):
            self.messages = [
                Message(role=msg["role"], text=msg["text"]) 
                for msg in self.messages
            ]
    
    def get_user_messages(self) -> List[Message]:
        """Get all user messages"""
        return [msg for msg in self.messages if msg.is_user_message()]
    
    def get_bot_messages(self) -> List[Message]:
        """Get all bot messages"""
        return [msg for msg in self.messages if msg.is_bot_message()]
    
    def get_conversation_text(self) -> str:
        """Get full conversation as single text"""
        return " ".join([msg.text for msg in self.messages])
    
    def get_user_text(self) -> str:
        """Get all user messages as single text"""
        user_messages = self.get_user_messages()
        return " ".join([msg.text for msg in user_messages])
    
    def get_message_count(self) -> int:
        """Get total number of messages"""
        return len(self.messages)
    
    def get_turn_count(self) -> int:
        """Get number of conversation turns (user-bot pairs)"""
        return len(self.get_user_messages())
    
    def get_conversation_length(self) -> int:
        """Get total word count of conversation"""
        return sum([msg.get_word_count() for msg in self.messages])

@dataclass
class Dataset:
    """Represents a dataset of conversations"""
    conversations: List[Conversation]
    name: str = "dataset"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Conversation:
        return self.conversations[idx]
    
    def get_labels(self) -> List[str]:
        """Get all intent labels"""
        return [conv.intent_label for conv in self.conversations]
    
    def get_unique_labels(self) -> List[str]:
        """Get unique intent labels"""
        return list(set(self.get_labels()))
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels"""
        labels = self.get_labels()
        return {label: labels.count(label) for label in set(labels)}
    
    def filter_by_label(self, label: str) -> 'Dataset':
        """Filter conversations by intent label"""
        filtered_convs = [conv for conv in self.conversations if conv.intent_label == label]
        return Dataset(
            conversations=filtered_convs,
            name=f"{self.name}_filtered_{label}",
            metadata=self.metadata.copy()
        )
    
    def sample(self, n: int, random_state: Optional[int] = None) -> 'Dataset':
        """Sample n conversations randomly"""
        import random
        if random_state is not None:
            random.seed(random_state)
        
        sampled_convs = random.sample(self.conversations, min(n, len(self.conversations)))
        return Dataset(
            conversations=sampled_convs,
            name=f"{self.name}_sampled_{n}",
            metadata=self.metadata.copy()
        )

@dataclass 
class ClassificationResult:
    """Represents classification result for a single conversation"""
    conversation_id: str
    true_label: str
    predicted_label: str
    prediction_confidence: float = 0.0
    prediction_probabilities: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    
    def is_correct(self) -> bool:
        """Check if prediction is correct"""
        return self.true_label == self.predicted_label

@dataclass
class EvaluationResults:
    """Represents evaluation results across multiple conversations"""
    results: List[ClassificationResult]
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[Any] = None
    classification_report: Optional[str] = None
    
    def get_accuracy(self) -> float:
        """Calculate accuracy"""
        if not self.results:
            return 0.0
        correct = sum([1 for r in self.results if r.is_correct()])
        return correct / len(self.results)
    
    def get_predictions(self) -> List[str]:
        """Get all predictions"""
        return [r.predicted_label for r in self.results]
    
    def get_true_labels(self) -> List[str]:
        """Get all true labels"""
        return [r.true_label for r in self.results]