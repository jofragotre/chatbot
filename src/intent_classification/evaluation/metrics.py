"""
Evaluation metrics for intent classification
"""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.preprocessing import LabelBinarizer

class IntentClassificationMetrics:
    """Calculate various metrics for intent classification"""
    
    @staticmethod
    def calculate_metrics(y_true: List[str], 
                         y_pred: List[str], 
                         y_proba: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Get unique labels
        labels = sorted(list(set(y_true + y_pred)))
        
        # Per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            }
        
        # Averaged metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall) 
        macro_f1 = np.mean(f1)
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class': per_class_metrics,
            'labels': labels
        }
        
        # Add AUC scores if probabilities available
        if y_proba is not None:
            try:
                auc_scores = IntentClassificationMetrics._calculate_auc_scores(
                    y_true, y_proba, labels
                )
                metrics.update(auc_scores)
                
                # Add PR-AUC scores
                pr_auc_scores = IntentClassificationMetrics._calculate_pr_auc_scores(
                    y_true, y_proba, labels
                )
                metrics.update(pr_auc_scores)
                
            except Exception as e:
                print(f"Warning: Could not calculate AUC scores: {e}")
        
        return metrics
    
    @staticmethod
    def _calculate_auc_scores(y_true: List[str], 
                            y_proba: List[Dict[str, float]], 
                            labels: List[str]) -> Dict[str, float]:
        """Calculate ROC-AUC scores"""
        
        # Convert to binary format for AUC calculation
        lb = LabelBinarizer()
        y_true_binary = lb.fit_transform(y_true)
        
        # Convert probabilities to matrix
        y_proba_matrix = np.zeros((len(y_proba), len(labels)))
        for i, proba_dict in enumerate(y_proba):
            for j, label in enumerate(labels):
                y_proba_matrix[i, j] = proba_dict.get(label, 0.0)
        
        auc_scores = {}
        
        if len(labels) == 2:
            # Binary classification
            auc_scores['roc_auc'] = roc_auc_score(y_true_binary, y_proba_matrix[:, 1])
        else:
            # Multi-class classification
            try:
                # Macro-averaged AUC
                auc_scores['roc_auc_macro'] = roc_auc_score(
                    y_true_binary, y_proba_matrix, average='macro', multi_class='ovr'
                )
                # Weighted AUC
                auc_scores['roc_auc_weighted'] = roc_auc_score(
                    y_true_binary, y_proba_matrix, average='weighted', multi_class='ovr'
                )
            except ValueError:
                # Fallback for cases where AUC can't be calculated
                pass
        
        return auc_scores
    
    @staticmethod
    def _calculate_pr_auc_scores(y_true: List[str], 
                               y_proba: List[Dict[str, float]], 
                               labels: List[str]) -> Dict[str, float]:
        """Calculate Precision-Recall AUC scores"""
        
        # Convert to binary format
        lb = LabelBinarizer()
        y_true_binary = lb.fit_transform(y_true)
        
        # Convert probabilities to matrix
        y_proba_matrix = np.zeros((len(y_proba), len(labels)))
        for i, proba_dict in enumerate(y_proba):
            for j, label in enumerate(labels):
                y_proba_matrix[i, j] = proba_dict.get(label, 0.0)
        
        pr_auc_scores = {}
        
        if len(labels) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(
                y_true_binary, y_proba_matrix[:, 1]
            )
            pr_auc_scores['pr_auc'] = auc(recall, precision)
        else:
            # Multi-class classification - calculate per-class PR-AUC
            per_class_pr_auc = []
            class_supports = []
            
            for i, label in enumerate(labels):
                try:
                    # One-vs-rest for each class
                    y_true_class = (y_true_binary[:, i] if y_true_binary.ndim > 1 
                                  else (y_true_binary == i).astype(int))
                    y_proba_class = y_proba_matrix[:, i]
                    
                    precision, recall, _ = precision_recall_curve(
                        y_true_class, y_proba_class
                    )
                    
                    if len(np.unique(y_true_class)) > 1:  # Only if class exists
                        class_pr_auc = auc(recall, precision)
                        per_class_pr_auc.append(class_pr_auc)
                        class_supports.append(np.sum(y_true_class))
                    else:
                        per_class_pr_auc.append(0.0)
                        class_supports.append(0)
                        
                except Exception:
                    per_class_pr_auc.append(0.0)
                    class_supports.append(0)
            
            # Macro-averaged PR-AUC
            if per_class_pr_auc:
                pr_auc_scores['pr_auc_macro'] = np.mean(per_class_pr_auc)
                
                # Weighted PR-AUC
                if sum(class_supports) > 0:
                    pr_auc_scores['pr_auc_weighted'] = np.average(
                        per_class_pr_auc, weights=class_supports
                    )
        
        return pr_auc_scores
    
    @staticmethod
    def get_confusion_matrix(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None):
        """Get confusion matrix"""
        if labels is None:
            labels = sorted(list(set(y_true + y_pred)))
        
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def get_classification_report(y_true: List[str], y_pred: List[str]) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, zero_division=0)