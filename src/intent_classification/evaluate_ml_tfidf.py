"""
Evaluate the TF-IDF + Classical ML classifier
- Matches the structure/outputs of evaluate_rule_classifier.py
"""
import sys
sys.path.append(".")

import json
import numpy as np

from intent_classification.data import ConversationLoader
from intent_classification.evaluation import ClassifierEvaluator
from intent_classification.utils import setup_logging, print_metrics_summary, sanitize_for_json
from classifiers.ml_tfidf import TfidfMLClassifier


def _maybe_print_top_features(clf: TfidfMLClassifier, top_k: int = 8):
    print("\n=== Top Indicative Features (per class, if available) ===")
    try:
        vec = clf.vectorizer
        model = clf.clf
        le = clf.label_encoder
        if vec is None or model is None or not hasattr(model, "coef_"):
            print("(Not available for this model)")
            return

        feature_names = np.array(vec.get_feature_names_out())
        coefs = model.coef_
        classes = le.classes_

        for idx, class_name in enumerate(classes):
            coef = coefs[idx]
            top_pos_idx = np.argsort(coef)[-top_k:][::-1]
            print(f"\nClass '{class_name}':")
            for j in top_pos_idx:
                print(f"  {feature_names[j]:<25}  weight={coef[j]:.3f}")
    except Exception as e:
        print(f"(Could not extract features: {e})")


def main():
    setup_logging(level="INFO")

    print("=== TF-IDF + Classical ML Intent Classifier Evaluation ===\n")

    # Load
    print("Loading dataset...")
    dataset = ConversationLoader.load_jsonl("../data/synth/synthetic_conversations_large.jsonl")
    print(f"Loaded {len(dataset)} conversations")
    print(f"Label distribution: {dataset.get_label_distribution()}")

    # Split (train/val/test for consistency)
    train_data, val_data, test_data = ConversationLoader.train_test_split(
        dataset, test_size=0.2, validation_size=0.1, random_state=0, stratify=True
    )
    print(f"\nTrain size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    # Configure classifier
    config = {
        "text_source": "full",
        "ngram_range": (1, 2),
        "max_features": 5000,
        "use_structural_features": True,
        "model_type": "svm",
        "C": 2.0,
        "class_weight": "balanced",
        "calibrate": True,
        "random_state": 0,
    }
    clf = TfidfMLClassifier(config=config)

    # Train
    print("\nTraining...")
    train_info = clf.train(train_data, validation_dataset=val_data)
    compact_train = {k: round(v, 4) if isinstance(v, float) else v for k, v in train_info.items()}
    print("Train summary:", compact_train)

    # Evaluate
    print("\nEvaluating on test set...")
    evaluator = ClassifierEvaluator()
    eval_res = evaluator.evaluate(clf, test_data)

    # Metrics
    print_metrics_summary(eval_res.metrics)

    # Example predictions
    print("\n=== Example Predictions ===")
    for i, result in enumerate(eval_res.results[:5]):
        print(f"\nConversation {i+1}:")
        print(f"True: {result.true_label} | Predicted: {result.predicted_label} | Confidence: {result.prediction_confidence:.3f}")
        conv = test_data.conversations[i]
        for msg in conv.messages[:3]:
            print(f"  {msg.role}: {msg.text}")
        if len(conv.messages) > 3:
            print(f"  ... ({len(conv.messages)-3} more messages)")

    # Classification report
    print(f"\n=== Classification Report ===")
    print(eval_res.classification_report)

    # Optional: show top features
    _maybe_print_top_features(clf, top_k=8)

    # Save detailed results
    results_summary = {
        "metrics": eval_res.metrics,
        "training_summary": compact_train,
        "model_info": clf.get_model_info(),
    }
    with open("ml_tfidf_results.json", "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results_summary), f, indent=2)
    print("\nDetailed results saved to: ml_tfidf_results.json")

    
    vec = clf.vectorizer
    le = clf.label_encoder
    tfidf_names = np.array(vec.get_feature_names_out())
    struct_names = np.array([
        "total_msgs","turns","n_user_msgs","n_bot_msgs","total_words","user_ratio",
        "avg_user_len","q_marks","ex_marks","digits","has_month","euro","price_words",
    ])
    feature_names = (np.concatenate([tfidf_names, struct_names])
                    if clf.config.get("use_structural_features", True)
                    else tfidf_names)
    sk_est = clf.clf

    # First try the direct estimator attribute
    base = None
    if hasattr(sk_est, 'estimator') and hasattr(sk_est.estimator, 'coef_'):
        base = sk_est.estimator
        print("Found coefficients in direct estimator")
    elif hasattr(sk_est, 'calibrated_classifiers_') and len(sk_est.calibrated_classifiers_) > 0:
        # Try the first calibrated classifier
        first_cal = sk_est.calibrated_classifiers_[0] 
        if hasattr(first_cal, 'estimator') and hasattr(first_cal.estimator, 'coef_'):
            base = first_cal.estimator
            print("Found coefficients in calibrated classifier's estimator")

    if base is not None and hasattr(base, 'coef_'):
        W = base.coef_  # shape: (n_classes, n_features)
        classes = le.classes_
        
        def top_for_class(k, topn=10):
            w = W[k]
            pos = np.argsort(w)[-topn:][::-1]
            neg = np.argsort(w)[:topn]
            return list(zip(feature_names[pos], w[pos])), list(zip(feature_names[neg], w[neg]))

        for k, cls_name in enumerate(classes):
            pos, neg = top_for_class(k, 10)
            print(f"\nClass {cls_name} — top positive:")
            for f, w in pos:
                print(f"  {f:30s} {w: .4f}")
            print("Top negative:")
            for f, w in neg:
                print(f"  {f:30s} {w: .4f}")
    else:
        print("Could not find LinearSVC coefficients in the calibrated classifier structure")
        # Let's try one more debugging step
        print("Trying to access sk_est.estimator directly...")
        if hasattr(sk_est, 'estimator'):
            print(f"sk_est.estimator type: {type(sk_est.estimator)}")
            print(f"sk_est.estimator attributes: {[a for a in dir(sk_est.estimator) if not a.startswith('_')]}")


if __name__ == "__main__":
    main()