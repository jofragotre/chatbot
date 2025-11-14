"""
Base feature extraction classes
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

from .data_structures import Conversation, Message

class FeatureExtractor(ABC):
    """Abstract base class for feature extraction"""
    
    @abstractmethod
    def fit(self, conversations: List[Conversation]) -> 'FeatureExtractor':
        """Fit feature extractor on conversations"""
        pass
    
    @abstractmethod
    def transform(self, conversations: List[Conversation]) -> np.ndarray:
        """Transform conversations to feature vectors"""
        pass
    
    def fit_transform(self, conversations: List[Conversation]) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(conversations).transform(conversations)

class ConversationFeatureExtractor(FeatureExtractor):
    """Extract various features from conversations"""
    
    def __init__(self, 
                 use_text_features: bool = True,
                 use_structural_features: bool = True,
                 use_linguistic_features: bool = True,
                 tfidf_params: Dict[str, Any] = None):
        
        self.use_text_features = use_text_features
        self.use_structural_features = use_structural_features
        self.use_linguistic_features = use_linguistic_features
        
        # Text vectorizer
        if self.use_text_features:
            tfidf_params = tfidf_params or {
                'max_features': 5000,
                'ngram_range': (1, 2),
                'stop_words': 'english',
                'lowercase': True
            }
            self.tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
        
        self.is_fitted = False
    
    def fit(self, conversations: List[Conversation]) -> 'ConversationFeatureExtractor':
        """Fit feature extractor"""
        if self.use_text_features:
            # Get all conversation texts for TF-IDF
            texts = [conv.get_user_text() for conv in conversations]
            self.tfidf_vectorizer.fit(texts)
        
        self.is_fitted = True
        return self
    
    def transform(self, conversations: List[Conversation]) -> np.ndarray:
        """Transform conversations to feature vectors"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        all_features = []
        
        for conv in conversations:
            features = []
            
            # Text features (TF-IDF)
            if self.use_text_features:
                user_text = conv.get_user_text()
                tfidf_features = self.tfidf_vectorizer.transform([user_text]).toarray()[0]
                features.extend(tfidf_features)
            
            # Structural features
            if self.use_structural_features:
                structural_features = self._extract_structural_features(conv)
                features.extend(structural_features)
            
            # Linguistic features
            if self.use_linguistic_features:
                linguistic_features = self._extract_linguistic_features(conv)
                features.extend(linguistic_features)
            
            all_features.append(features)
        
        return np.array(all_features)
    
    def _extract_structural_features(self, conv: Conversation) -> List[float]:
        """Extract structural features from conversation"""
        features = []
        
        # Basic counts
        features.append(conv.get_message_count())           # Total messages
        features.append(conv.get_turn_count())              # Conversation turns
        features.append(len(conv.get_user_messages()))      # User messages
        features.append(len(conv.get_bot_messages()))       # Bot messages
        features.append(conv.get_conversation_length())     # Total words
        
        # Ratios
        total_msgs = conv.get_message_count()
        if total_msgs > 0:
            features.append(len(conv.get_user_messages()) / total_msgs)  # User message ratio
        else:
            features.append(0)
        
        # Message lengths
        user_messages = conv.get_user_messages()
        if user_messages:
            user_lengths = [msg.get_word_count() for msg in user_messages]
            features.append(np.mean(user_lengths))          # Avg user message length
            features.append(np.max(user_lengths))           # Max user message length
            features.append(np.min(user_lengths))           # Min user message length
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_linguistic_features(self, conv: Conversation) -> List[float]:
        """Extract linguistic features from conversation"""
        features = []
        user_text = conv.get_user_text().lower()
        
        # Booking intent keywords
        booking_keywords = ['book', 'reserve', 'confirm', 'proceed', 'yes', 'ok', 'take it']
        exploration_keywords = ['what', 'where', 'when', 'how', 'tell me', 'info', 'about']
        price_keywords = ['price', 'cost', 'rate', 'fee', 'charge', 'euro', '€', '$']
        urgency_keywords = ['urgent', 'asap', 'quickly', 'now', 'today', 'tonight']
        
        # Count keyword occurrences
        features.append(sum([user_text.count(kw) for kw in booking_keywords]))
        features.append(sum([user_text.count(kw) for kw in exploration_keywords]))
        features.append(sum([user_text.count(kw) for kw in price_keywords]))
        features.append(sum([user_text.count(kw) for kw in urgency_keywords]))
        
        # Question marks and exclamation marks
        features.append(user_text.count('?'))
        features.append(user_text.count('!'))
        
        # Dates (simple regex)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
        features.append(len(re.findall(date_pattern, user_text, re.IGNORECASE)))
        
        # Numbers (could indicate dates, room numbers, etc.)
        number_pattern = r'\b\d+\b'
        features.append(len(re.findall(number_pattern, user_text)))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        names = []
        
        if self.use_text_features and hasattr(self, 'tfidf_vectorizer'):
            names.extend([f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()])
        
        if self.use_structural_features:
            names.extend([
                'total_messages', 'turn_count', 'user_messages', 'bot_messages',
                'total_words', 'user_msg_ratio', 'avg_user_msg_len', 
                'max_user_msg_len', 'min_user_msg_len'
            ])
        
        if self.use_linguistic_features:
            names.extend([
                'booking_keywords', 'exploration_keywords', 'price_keywords', 
                'urgency_keywords', 'question_marks', 'exclamation_marks',
                'date_mentions', 'number_mentions'
            ])
        
        return names