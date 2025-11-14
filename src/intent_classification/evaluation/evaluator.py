"""
Evaluation framework for intent classifiers
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from ..base.classifier import IntentClassifier
from ..base.data_structures import Dataset, ClassificationResult, EvaluationResults
from .metrics import IntentClassificationMetrics

logger = logging.getLogger(__name__)

class ClassifierEvaluator:
    """Evaluate intent classifiers"""
    
    def __init__(self):
        self.metrics_calculator = IntentClassificationMetrics()
    
    def evaluate(self, classifier: IntentClassifier, test_dataset: Dataset) -> EvaluationResults:
        """Evaluate classifier on test dataset"""
        
        if not classifier.is_trained:
            raise ValueError("Classifier must be trained before evaluation")
        
        logger.info(f"Evaluating classifier on {len(test_dataset)} conversations")
        
        # Get predictions and probabilities
        predictions = classifier.predict(test_dataset.conversations)
        probabilities = classifier.predict_proba(test_dataset.conversations)
        
        # Create classification results
        results = []
        for i, conv in enumerate(test_dataset.conversations):
            result = ClassificationResult(
                conversation_id=conv.session_id,
                true_label=conv.intent_label,
                predicted_label=predictions[i],
                prediction_confidence=max(probabilities[i].values()) if probabilities[i] else 0.0,
                prediction_probabilities=probabilities[i]
            )
            results.append(result)
        
        # Calculate metrics
        true_labels = [r.true_label for r in results]
        pred_labels = [r.predicted_label for r in results]
        pred_probabilities = [r.prediction_probabilities for r in results]
        
        metrics = self.metrics_calculator.calculate_metrics(
            true_labels, pred_labels, pred_probabilities
        )
        
        # Get confusion matrix and classification report
        confusion_matrix = self.metrics_calculator.get_confusion_matrix(true_labels, pred_labels)
        classification_report = self.metrics_calculator.get_classification_report(true_labels, pred_labels)
        
        return EvaluationResults(
            results=results,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            classification_report=classification_report
        )
    
    def cross_validate(self, 
                      classifier: IntentClassifier, 
                      dataset: Dataset,
                      cv_folds: int = 5,
                      scoring: str = 'accuracy') -> Dict[str, Any]:
        """Perform cross-validation"""
        
        # For now, this is a simplified version
        # In practice, you'd need to implement proper CV for the conversation data
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # This would need to be implemented based on the specific classifier type
        # For now, return a placeholder
        return {
            'cv_scores': [],
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'message': 'Cross-validation not yet implemented for conversation classifiers'
        }
    
    def evaluate_multiple_classifiers(self, 
                                    classifiers: Dict[str, IntentClassifier],
                                    test_dataset: Dataset) -> Dict[str, EvaluationResults]:
        """Evaluate multiple classifiers on the same dataset"""
        
        results = {}
        
        for name, classifier in classifiers.items():
            logger.info(f"Evaluating classifier: {name}")
            try:
                eval_result = self.evaluate(classifier, test_dataset)
                results[name] = eval_result
            except Exception as e:
                logger.error(f"Failed to evaluate classifier {name}: {e}")
                continue
        
        return results
    
    def compare_classifiers(self, 
                          evaluation_results: Dict[str, EvaluationResults],
                          metric: str = 'accuracy') -> Dict[str, Any]:
        """Compare multiple classifier evaluation results"""
        
        comparison = {}
        
        for name, result in evaluation_results.items():
            if metric in result.metrics:
                comparison[name] = result.metrics[metric]
            else:
                comparison[name] = None
        
        # Sort by performance
        valid_results = {k: v for k, v in comparison.items() if v is not None}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'metric': metric,
            'results': comparison,
            'ranking': sorted_results,
            'best_classifier': sorted_results[0][0] if sorted_results else None,
            'best_score': sorted_results[0][1] if sorted_results else None
        }