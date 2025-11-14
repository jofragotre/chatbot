"""
Evaluate the TF-IDF + Classical ML classifier
"""
import sys

sys.path.append(".")

from intent_classification.data import ConversationLoader
from intent_classification.utils import (
    setup_logging,
    print_metrics_summary,
)
from intent_classification.evaluation import ClassifierEvaluator
from classifiers.ml_tfidf import TfidfMLClassifier


def main():
    setup_logging(level="INFO")

    print("=== TF-IDF + Classical ML Intent Classifier ===\n")

    # Load your dataset (synthetic or real)
    dataset = ConversationLoader.load_jsonl("../data/synth/synthetic_conversations_v2.jsonl")
    print(f"Loaded {len(dataset)} conversations")
    print(f"Label distribution: {dataset.get_label_distribution()}")

    # Split
    train_data, val_data, test_data = ConversationLoader.train_test_split(
        dataset, test_size=0.2, validation_size=0.1, random_state=42, stratify=True
    )

    # Configure classifier
    config = {
        "text_source": "user",  # try 'full' for combined user+bot text
        "ngram_range": (1, 2),
        "max_features": 12000,
        "use_structural_features": True,
        "model_type": "logreg",  # try 'svm' with 'calibrate': True
        "C": 2.0,
        "class_weight": "balanced",
        "calibrate": True,
        "random_state": 42,
    }

    clf = TfidfMLClassifier(config=config)

    # Train
    train_info = clf.train(train_data, validation_dataset=val_data)
    print("\nTrain summary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in train_info.items()})

    # Evaluate on test set with the framework
    evaluator = ClassifierEvaluator()
    eval_res = evaluator.evaluate(clf, test_data)

    print_metrics_summary(eval_res.metrics)
    print("\n=== Classification Report ===")
    print(eval_res.classification_report)

    # Optional: save model
    # clf.save("artifacts/tfidf_ml_classifier.pkl")


if __name__ == "__main__":
    main()