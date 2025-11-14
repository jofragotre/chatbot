"""
Quick test of the rule-based classifier with sample data
"""
import sys
sys.path.append('.')

from intent_classification.base.data_structures import Conversation, Message
from classifiers.rule_based import RuleBasedClassifier

def create_test_conversations():
    """Create some test conversations manually"""
    
    # High intent conversation
    high_conv = Conversation(
        session_id="test_high",
        messages=[
            Message("user", "Hi, I want to book a room for Oct 15-17"),
            Message("bot", "Great! We have a Deluxe room for €180/night. Proceed?"),
            Message("user", "Yes, book it please. My name is John Smith"),
            Message("bot", "Perfect! I'll need your email to confirm.")
        ],
        intent_label="high"
    )
    
    # Medium intent conversation  
    medium_conv = Conversation(
        session_id="test_medium",
        messages=[
            Message("user", "What's the rate for weekend stays?"),
            Message("bot", "Weekend rates start at €150 for Standard rooms"),
            Message("user", "What's included in the breakfast package?"),
            Message("bot", "Continental breakfast with local specialties")
        ],
        intent_label="medium"
    )
    
    # Low intent conversation
    low_conv = Conversation(
        session_id="test_low", 
        messages=[
            Message("user", "Hi, what's nearby your hotel?"),
            Message("bot", "We're near the city center with shops and restaurants"),
            Message("user", "Any weekend deals?"),
            Message("bot", "We have packages starting at €199")
        ],
        intent_label="low"
    )
    
    # Abandoned conversation
    abandoned_conv = Conversation(
        session_id="test_abandoned",
        messages=[
            Message("user", "I need a room for tonight"),
            Message("bot", "Single room available for €120. Shall I book it?"),
            Message("user", "Yes please"),
            Message("bot", "I'll need your card details to guarantee"),
            Message("user", "Actually, I'll finish this later")
        ],
        intent_label="abandoned"
    )
    
    return [high_conv, medium_conv, low_conv, abandoned_conv]

def main():
    print("=== Quick Rule-Based Classifier Test ===\n")
    
    # Create test conversations
    test_conversations = create_test_conversations()
    
    # Initialize classifier
    classifier = RuleBasedClassifier()
    
    # Test predictions
    predictions = classifier.predict(test_conversations)
    probabilities = classifier.predict_proba(test_conversations)
    
    # Show results
    for i, conv in enumerate(test_conversations):
        print(f"Conversation {i+1} ({conv.session_id}):")
        print(f"True label: {conv.intent_label}")
        print(f"Predicted: {predictions[i]}")
        print(f"Probabilities: {probabilities[i]}")
        print(f"Correct: {'✓' if predictions[i] == conv.intent_label else '✗'}")
        print(f"Conversation: {conv.get_user_text()[:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()