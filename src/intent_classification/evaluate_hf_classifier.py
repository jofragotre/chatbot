"""
Evaluate the Hugging Face transformer-based intent classifier
- Matches the structure/outputs of evaluate_rule_classifier.py
"""
import sys
sys.path.append(".")

import json

from intent_classification.data import ConversationLoader
from intent_classification.evaluation import ClassifierEvaluator
from intent_classification.utils import setup_logging, print_metrics_summary
from classifiers.hf_sequence_classifier import HFSequenceClassifier


def main():
    setup_logging(level="INFO")

    print("=== HF Transformer Intent Classifier Evaluation ===\n")

    # Load dataset
    print("Loading dataset...")
    dataset = ConversationLoader.load_jsonl("synthetic_conversations.jsonl")
    print(f"Loaded {len(dataset)} conversations")
    print(f"Label distribution: {dataset.get_label_distribution()}")

    # Split data (train/val/test)
    train_data, val_data, test_data = ConversationLoader.train_test_split(
        dataset, test_size=0.2, validation_size=0.1, random_state=42, stratify=True
    )
    print(f"\nTrain size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    # Initialize classifier
    print("\nInitializing HF classifier...")
    config = {
        "model_name": "distilbert-base-uncased",  # try 'microsoft/deberta-v3-base' if you have GPU
        "text_source": "full",  # 'user' | 'full' (full can capture bot-side booking steps)
        "max_length": 512,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "early_stopping_patience": 2,
        "output_dir": "artifacts/hf_intent_cls",
        "fp16": True,
    }
    classifier = HFSequenceClassifier(config=config)

    # Train
    print("\nTraining...")
    training_results = classifier.train(train_data, validation_dataset=val_data)
    compact_train = {k: round(v, 4) if isinstance(v, float) else v for k, v in training_results.items()}
    print("Train summary:", compact_train)

    # Evaluate on test
    print("\nEvaluating on test set...")
    evaluator = ClassifierEvaluator()
    eval_results = evaluator.evaluate(classifier, test_data)

    # Metrics summary
    print_metrics_summary(eval_results.metrics)

    # Example predictions
    print("\n=== Example Predictions ===")
    for i, result in enumerate(eval_results.results[:5]):
        print(f"\nConversation {i+1}:")
        print(f"True: {result.true_label} | Predicted: {result.predicted_label} | Confidence: {result.prediction_confidence:.3f}")
        conv = test_data.conversations[i]
        for msg in conv.messages[:3]:
            print(f"  {msg.role}: {msg.text}")
        if len(conv.messages) > 3:
            print(f"  ... ({len(conv.messages)-3} more messages)")

    # Classification report
    print(f"\n=== Classification Report ===")
    print(eval_results.classification_report)

    # Save detailed results
    results_summary = {
        "metrics": eval_results.metrics,
        "training_summary": compact_train,
        "model_info": classifier.get_model_info(),
    }
    with open("hf_classifier_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print("\nDetailed results saved to: hf_classifier_results.json")


if __name__ == "__main__":
    main()