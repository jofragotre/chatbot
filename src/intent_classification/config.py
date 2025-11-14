"""
Configuration for intent classification
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ClassificationConfig:
    """Configuration for intent classification"""
    
    # Data configuration
    data_path: str = "data/conversations.jsonl"
    test_size: float = 0.2
    validation_size: Optional[float] = 0.1
    random_state: int = 42
    
    # Feature extraction configuration
    use_text_features: bool = True
    use_structural_features: bool = True
    use_linguistic_features: bool = True
    
    # TF-IDF parameters
    tfidf_params: Dict[str, Any] = None
    
    # Evaluation configuration
    cross_validation_folds: int = 5
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        if self.tfidf_params is None:
            self.tfidf_params = {
                'max_features': 5000,
                'ngram_range': (1, 2),
                'stop_words': 'english',
                'lowercase': True,
                'min_df': 2,
                'max_df': 0.95
            }
        
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                'accuracy', 'macro_f1', 'weighted_f1', 
                'macro_precision', 'macro_recall'
            ]

# Intent label mappings
INTENT_LABELS = {
    'low': 'Low Intent (Exploring)',
    'medium': 'Medium Intent (Evaluating)', 
    'high': 'High Intent (Actioning)',
    'abandoned': 'Abandoned Intent'
}

LABEL_MAPPING = {
    0: 'low',
    1: 'medium', 
    2: 'high',
    3: 'abandoned'
}