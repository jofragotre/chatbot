"""
Evaluate the HF zero-shot pipeline intent classifier
- Matches the structure/outputs of evaluate_rule_classifier.py
"""
import sys
sys.path.append(".")

import json

from intent_classification.data import ConversationLoader
from intent_classification.evaluation import ClassifierEvaluator
from intent_classification.utils import setup_logging, print_metrics_summary, sanitize_for_json
from classifiers.hf_pipeline_classifier import HFZeroShotPipelineClassifier


def main():
    setup_logging(level="INFO")

    print("=== HF Zero-Shot Pipeline Intent Classifier Evaluation ===\n")

    # Load dataset
    print("Loading dataset...")
    dataset = ConversationLoader.load_jsonl("../data/synth/synthetic_conversations_v3.jsonl")
    print(f"Loaded {len(dataset)} conversations")
    print(f"Label distribution: {dataset.get_label_distribution()}")

    # Split (train/val/test for consistency with other scripts)
    train_data, val_data, test_data = ConversationLoader.train_test_split(
        dataset, test_size=0.2, validation_size=0.1, random_state=0, stratify=True
    )
    print(f"\nTrain size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    # Initialize classifier
    print("\nInitializing HF zero-shot classifier...")
    config = {
        "model_name": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "text_source": "user", 
        "hypothesis_template": "The user in this conversation is {}.",
    }
    classifier = HFZeroShotPipelineClassifier(config=config)

    # "Train" (initialize pipeline)
    print("\nSetting up pipeline (no training)...")
    training_results = classifier.train(train_data, validation_dataset=val_data)
    compact_train = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in training_results.items()
    }
    print("Train summary:", compact_train)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = ClassifierEvaluator()
    eval_results = evaluator.evaluate(classifier, test_data)

    # Metrics summary
    print_metrics_summary(eval_results.metrics)

    # Example predictions
    print("\n=== Example Predictions ===")
    for i, result in enumerate(eval_results.results[:5]):
        print(f"\nConversation {i+1}:")
        print(
            f"True: {result.true_label} | Predicted: {result.predicted_label} | "
            f"Confidence: {result.prediction_confidence:.3f}"
        )
        conv = test_data.conversations[i]
        for msg in conv.messages: #[:3]:
            print(f"  {msg.role}: {msg.text}")

    # Classification report
    print(f"\n=== Classification Report ===")
    print(eval_results.classification_report)

    # Save detailed results
    results_summary = {
        "metrics": eval_results.metrics,
        "training_summary": compact_train,
        "model_info": classifier.get_model_info(),
    }
    with open("hf_pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results_summary), f, indent=2)
    print("\nDetailed results saved to: hf_pipeline_results.json")


if __name__ == "__main__":
    main()