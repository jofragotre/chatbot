"""
Base classifier interface for intent classification
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle
from pathlib import Path
import logging

from .data_structures import Conversation, Dataset, ClassificationResult, EvaluationResults

logger = logging.getLogger(__name__)

class IntentClassifier(ABC):
    """Abstract base class for all intent classifiers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_trained = False
        self.label_encoder = None
        self.feature_extractor = None
        self.model = None
        
    @abstractmethod
    def train(self, train_dataset: Dataset, validation_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Train the classifier
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Optional validation dataset
            
        Returns:
            Training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, conversations: List[Conversation]) -> List[str]:
        """
        Predict intent labels for conversations
        
        Args:
            conversations: List of conversations to classify
            
        Returns:
            List of predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        """
        Predict intent probabilities for conversations
        
        Args:
            conversations: List of conversations to classify
            
        Returns:
            List of probability dictionaries {label: probability}
        """
        pass
    
    def predict_single(self, conversation: Conversation) -> str:
        """Predict intent for a single conversation"""
        return self.predict([conversation])[0]
    
    def predict_proba_single(self, conversation: Conversation) -> Dict[str, float]:
        """Predict probabilities for a single conversation"""
        return self.predict_proba([conversation])[0]
    
    def fit(self, train_dataset: Dataset, validation_dataset: Optional[Dataset] = None) -> 'IntentClassifier':
        """Sklearn-style fit method"""
        self.train(train_dataset, validation_dataset)
        return self
    
    def evaluate(self, test_dataset: Dataset) -> EvaluationResults:
        """
        Evaluate classifier on test dataset
        
        Args:
            test_dataset: Dataset to evaluate on
            
        Returns:
            Evaluation results
        """
        from ..evaluation.evaluator import ClassifierEvaluator
        evaluator = ClassifierEvaluator()
        return evaluator.evaluate(self, test_dataset)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save classifier to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'config': self.config,
            'is_trained': self.is_trained,
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'label_encoder': self.label_encoder,
            'classifier_type': self.__class__.__name__
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Classifier saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'IntentClassifier':
        """Load classifier from disk"""
        path = Path(path)
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create instance
        instance = cls(save_dict['config'])
        instance.is_trained = save_dict['is_trained']
        instance.model = save_dict['model']
        instance.feature_extractor = save_dict['feature_extractor']
        instance.label_encoder = save_dict['label_encoder']
        
        logger.info(f"Classifier loaded from {path}")
        return instance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'classifier_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config
        }

class EnsembleClassifier(IntentClassifier):
    """Base class for ensemble classifiers"""
    
    def __init__(self, classifiers: List[IntentClassifier], weights: Optional[List[float]] = None, config: Dict[str, Any] = None):
        super().__init__(config)
        self.classifiers = classifiers
        self.weights = weights or [1.0] * len(classifiers)
        
        if len(self.weights) != len(self.classifiers):
            raise ValueError("Number of weights must match number of classifiers")
    
    def train(self, train_dataset: Dataset, validation_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Train all classifiers in ensemble"""
        training_results = {}
        
        for i, classifier in enumerate(self.classifiers):
            logger.info(f"Training classifier {i+1}/{len(self.classifiers)}: {classifier.__class__.__name__}")
            result = classifier.train(train_dataset, validation_dataset)
            training_results[f'classifier_{i}'] = result
        
        self.is_trained = True
        return training_results
    
    def predict(self, conversations: List[Conversation]) -> List[str]:
        """Predict using weighted voting"""
        if not all(c.is_trained for c in self.classifiers):
            raise ValueError("All classifiers must be trained before prediction")
        
        # Get predictions from all classifiers
        all_predictions = []
        for classifier in self.classifiers:
            preds = classifier.predict(conversations)
            all_predictions.append(preds)
        
        # Weighted voting
        final_predictions = []
        for i in range(len(conversations)):
            votes = {}
            for j, classifier_preds in enumerate(all_predictions):
                pred = classifier_preds[i]
                weight = self.weights[j]
                votes[pred] = votes.get(pred, 0) + weight
            
            # Get prediction with highest weighted vote
            final_pred = max(votes, key=votes.get)
            final_predictions.append(final_pred)
        
        return final_predictions
    
    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        """Predict probabilities using weighted averaging"""
        if not all(c.is_trained for c in self.classifiers):
            raise ValueError("All classifiers must be trained before prediction")
        
        # Get probabilities from all classifiers
        all_probas = []
        for classifier in self.classifiers:
            probas = classifier.predict_proba(conversations)
            all_probas.append(probas)
        
        # Weighted averaging
        final_probas = []
        for i in range(len(conversations)):
            combined_proba = {}
            total_weight = 0
            
            for j, classifier_probas in enumerate(all_probas):
                weight = self.weights[j]
                total_weight += weight
                
                for label, prob in classifier_probas[i].items():
                    combined_proba[label] = combined_proba.get(label, 0) + prob * weight
            
            # Normalize
            for label in combined_proba:
                combined_proba[label] /= total_weight
            
            final_probas.append(combined_proba)
        
        return final_probas