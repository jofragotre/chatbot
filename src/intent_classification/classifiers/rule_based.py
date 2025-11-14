"""
Rule-based intent classifier
"""
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from base.classifier import IntentClassifier
from base.data_structures import Conversation, Dataset

logger = logging.getLogger(__name__)

class RuleBasedClassifier(IntentClassifier):
    """Simple rule-based intent classifier using keywords and patterns"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Define keyword patterns for each intent
        self.patterns = {
            'high': {
                'booking_commands': [
                    'book', 'reserve', 'confirm', 'proceed', 'yes please', 
                    'i want', 'i need', 'i\'ll take', 'go ahead', 'sounds good'
                ],
                'commitment_phrases': [
                    'name is', 'email is', 'credit card', 'payment',
                    'confirm booking', 'finalize', 'complete'
                ],
                'urgency': ['today', 'tonight', 'now', 'asap', 'urgent']
            },
            
            'medium': {
                'comparison_questions': [
                    'what\'s included', 'cancellation', 'policy', 'difference between',
                    'compare', 'options', 'breakfast included', 'free wifi'
                ],
                'specific_inquiries': [
                    'price for', 'cost for', 'rate for', 'available on',
                    'room type', 'check availability', 'what dates'
                ],
                'consideration_phrases': [
                    'let me think', 'i\'m considering', 'looking at',
                    'comparing', 'deciding between'
                ]
            },
            
            'low': {
                'general_questions': [
                    'what\'s nearby', 'tell me about', 'any deals', 'weekend package',
                    'location', 'area', 'attractions', 'restaurants'
                ],
                'browsing_behavior': [
                    'just looking', 'browsing', 'information about',
                    'curious about', 'wondering'
                ]
            },
            
            'abandoned': {
                'postponement': [
                    'i\'ll finish later', 'call back', 'think about it',
                    'let me check', 'get back to you', 'maybe later'
                ],
                'session_end': [
                    'hold the room', 'can you wait', 'pause',
                    'finish tomorrow', 'busy now'
                ]
            }
        }
        
        # Structural thresholds
        self.thresholds = {
            'min_messages_high': 3,      # Need some interaction for high intent
            'max_messages_low': 4,       # Too many messages suggest deeper interest
            'min_user_messages_medium': 2, # Medium needs comparison questions
            'min_words_commitment': 50   # High intent conversations tend to be longer
        }
        
        self.is_trained = True  # Rules don't need training
    
    def train(self, train_dataset: Dataset, validation_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Rule-based classifier doesn't need training, but we can analyze patterns"""
        logger.info("Analyzing training data for rule optimization...")
        
        # Analyze patterns in training data for potential rule improvements
        analysis = self._analyze_training_data(train_dataset)
        
        return {
            'training_method': 'rule_based',
            'rules_count': sum(len(patterns) for patterns in self.patterns.values()),
            'training_analysis': analysis
        }
    
    def predict(self, conversations: List[Conversation]) -> List[str]:
        """Predict intent labels for conversations"""
        predictions = []
        
        for conv in conversations:
            prediction = self._classify_conversation(conv)
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, conversations: List[Conversation]) -> List[Dict[str, float]]:
        """Predict probabilities for each intent"""
        probabilities = []
        
        for conv in conversations:
            proba = self._get_classification_probabilities(conv)
            probabilities.append(proba)
        
        return probabilities
    
    def _classify_conversation(self, conversation: Conversation) -> str:
        """Classify a single conversation"""
        user_text = conversation.get_user_text().lower()
        user_messages = conversation.get_user_messages()
        
        # Get scores for each intent
        scores = {}
        for intent in ['high', 'medium', 'low', 'abandoned']:
            scores[intent] = self._calculate_intent_score(conversation, intent)
        
        # Apply structural filters
        scores = self._apply_structural_filters(conversation, scores)
        
        # Return intent with highest score
        predicted_intent = max(scores, key=scores.get)
        
        logger.debug(f"Conversation {conversation.session_id}: scores={scores}, predicted={predicted_intent}")
        
        return predicted_intent
    
    def _calculate_intent_score(self, conversation: Conversation, intent: str) -> float:
        """Calculate score for a specific intent"""
        user_text = conversation.get_user_text().lower()
        score = 0.0
        
        intent_patterns = self.patterns[intent]
        
        # Count keyword matches
        for pattern_type, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in user_text)
            
            # Weight different pattern types
            if pattern_type in ['booking_commands', 'commitment_phrases']:
                score += matches * 3.0  # High weight for strong intent signals
            elif pattern_type in ['postponement', 'session_end']:
                score += matches * 2.5  # High weight for abandonment
            elif pattern_type in ['comparison_questions', 'specific_inquiries']:
                score += matches * 2.0  # Medium weight for evaluation
            else:
                score += matches * 1.0  # Base weight
        
        # Add structural bonuses
        if intent == 'high':
            # Longer conversations with specific details
            if conversation.get_conversation_length() > 50:
                score += 1.0
            if self._has_specific_details(conversation):
                score += 2.0
                
        elif intent == 'medium':
            # Multiple questions, comparison behavior
            question_count = conversation.get_user_text().count('?')
            score += min(question_count * 0.5, 2.0)
            
        elif intent == 'low':
            # Short, general conversations
            if conversation.get_message_count() <= 4:
                score += 1.0
                
        elif intent == 'abandoned':
            # Look for incomplete booking flows
            if self._suggests_incomplete_booking(conversation):
                score += 3.0
        
        return score
    
    def _apply_structural_filters(self, conversation: Conversation, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply structural constraints to scores"""
        filtered_scores = scores.copy()
        
        message_count = conversation.get_message_count()
        user_message_count = len(conversation.get_user_messages())
        
        # Apply thresholds
        if message_count < self.thresholds['min_messages_high']:
            filtered_scores['high'] *= 0.5  # Reduce high intent for very short conversations
            
        if message_count > self.thresholds['max_messages_low']:
            filtered_scores['low'] *= 0.3   # Reduce low intent for longer conversations
            
        if user_message_count < self.thresholds['min_user_messages_medium']:
            filtered_scores['medium'] *= 0.5  # Medium intent needs some back-and-forth
        
        return filtered_scores
    
    def _has_specific_details(self, conversation: Conversation) -> bool:
        """Check if conversation contains specific booking details"""
        user_text = conversation.get_user_text().lower()
        
        # Look for dates, numbers, names, emails
        patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # Dates
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',  # Month dates
            r'\b[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+\b',  # Emails
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',      # Names (simple pattern)
        ]
        
        for pattern in patterns:
            if re.search(pattern, user_text):
                return True
        
        return False
    
    def _suggests_incomplete_booking(self, conversation: Conversation) -> bool:
        """Check if conversation suggests an incomplete booking"""
        user_text = conversation.get_user_text().lower()
        bot_text = " ".join([msg.text.lower() for msg in conversation.get_bot_messages()])
        
        # Look for signs of abandonment
        abandonment_signals = [
            ('payment' in bot_text or 'card' in bot_text) and ('later' in user_text),
            'hold' in user_text and 'room' in user_text,
            len(conversation.messages) > 4 and conversation.messages[-1].role == 'bot'  # Bot had last word
        ]
        
        return any(abandonment_signals)
    
    def _get_classification_probabilities(self, conversation: Conversation) -> Dict[str, float]:
        """Get probability distribution across all intents"""
        # Get raw scores
        scores = {}
        for intent in ['high', 'medium', 'low', 'abandoned']:
            scores[intent] = self._calculate_intent_score(conversation, intent)
        
        # Apply structural filters
        scores = self._apply_structural_filters(conversation, scores)
        
        # Convert to probabilities using softmax-like normalization
        # Add small epsilon to avoid division by zero
        epsilon = 0.01
        total_score = sum(scores.values()) + epsilon * len(scores)
        
        probabilities = {}
        for intent, score in scores.items():
            probabilities[intent] = (score + epsilon) / total_score
        
        return probabilities
    
    def _analyze_training_data(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze training data to understand patterns"""
        analysis = {
            'total_conversations': len(dataset),
            'label_distribution': dataset.get_label_distribution(),
            'avg_lengths': {},
            'keyword_coverage': {}
        }
        
        # Analyze by label
        for label in dataset.get_unique_labels():
            label_convs = dataset.filter_by_label(label).conversations
            
            if label_convs:
                lengths = [conv.get_conversation_length() for conv in label_convs]
                analysis['avg_lengths'][label] = np.mean(lengths)
                
                # Check keyword coverage
                covered_convs = 0
                for conv in label_convs:
                    if self._calculate_intent_score(conv, label) > 0:
                        covered_convs += 1
                
                analysis['keyword_coverage'][label] = covered_convs / len(label_convs)
        
        return analysis
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get importance of different rule categories"""
        importance = {}
        
        for intent, patterns in self.patterns.items():
            for pattern_type, keywords in patterns.items():
                key = f"{intent}_{pattern_type}"
                # Simple importance based on number of keywords and their weights
                if pattern_type in ['booking_commands', 'commitment_phrases']:
                    importance[key] = len(keywords) * 3.0
                elif pattern_type in ['postponement', 'session_end']:
                    importance[key] = len(keywords) * 2.5
                elif pattern_type in ['comparison_questions', 'specific_inquiries']:
                    importance[key] = len(keywords) * 2.0
                else:
                    importance[key] = len(keywords) * 1.0
        
        return importance