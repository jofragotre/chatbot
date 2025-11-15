"""
Example script to evaluate the rule-based classifier
"""
import sys
sys.path.append('.')

from intent_classification.data import ConversationLoader
from intent_classification.evaluation import ClassifierEvaluator
from intent_classification.utils import setup_logging, print_metrics_summary
from classifiers.rule_based import RuleBasedClassifier
from utils import sanitize_for_json

def main():
    # Setup logging
    setup_logging(level="INFO")
    
    print("=== Rule-Based Intent Classifier Evaluation ===\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ConversationLoader.load_jsonl("../data/synth/synthetic_conversations_v3.jsonl")
    print(f"Loaded {len(dataset)} conversations")
    print(f"Label distribution: {dataset.get_label_distribution()}")
    
    # Split data
    train_data, test_data = ConversationLoader.train_test_split(
        dataset, 
        test_size=0.3, 
        random_state=0
    )
    
    print(f"\nTrain size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    
    # Create and "train" classifier
    print("\nInitializing rule-based classifier...")
    classifier = RuleBasedClassifier()
    
    # Train (analyze patterns)
    training_results = classifier.train(train_data)
    print(f"Training analysis: {training_results['training_analysis']}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = ClassifierEvaluator()
    eval_results = evaluator.evaluate(classifier, test_data)
    
    # Print results
    print_metrics_summary(eval_results.metrics)
    
    # Show some example predictions
    print("\n=== Example Predictions ===")
    for i, result in enumerate(eval_results.results[:5]):  # First 5 examples
        print(f"\nConversation {i+1}:")
        print(f"True: {result.true_label} | Predicted: {result.predicted_label} | Confidence: {result.prediction_confidence:.3f}")
        
        # Show the conversation
        conv = test_data.conversations[i]
        for msg in conv.messages[:3]:  # Show first 3 messages
            print(f"  {msg.role}: {msg.text}")
        if len(conv.messages) > 3:
            print(f"  ... ({len(conv.messages)-3} more messages)")
    
    # Show confusion matrix
    print(f"\n=== Classification Report ===")
    print(eval_results.classification_report)
    
    # Feature importance
    importance = classifier.get_feature_importance()
    print(f"\n=== Rule Importance (Top 10) ===")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, score in sorted_importance[:10]:
        print(f"{feature}: {score:.2f}")
    
    # Save detailed results
    import json
    results_summary = {
        'metrics': eval_results.metrics,
        'training_analysis': training_results['training_analysis'],
        'rule_importance': importance
    }
    
    with open('rule_classifier_results.json', 'w') as f:
        results_summary = sanitize_for_json(results_summary)
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: rule_classifier_results.json")

if __name__ == "__main__":
    main()