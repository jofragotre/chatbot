"""
Configuration settings for synthetic conversation generation
"""
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "Qwen/Qwen3-1.7B"
    device_map: str = "cuda:0"
    torch_dtype: str = "float16"
    max_new_tokens: int = 1024
    temperature: float = 0.85
    do_sample: bool = True
    top_p: float = 0.9

@dataclass
class GenerationConfig:
    """Data generation configuration"""
    conversations_per_intent: int = 50
    max_retries: int = 3
    batch_size: int = 8
    
    # Advanced generation probabilities
    use_persona_probability: float = 0.5
    use_few_shot_probability: float = 0.2
    
    # Intent distribution
    intent_weights: Dict[str, float] = None

# Enhanced hotel contexts with categories
HOTEL_CATEGORIES = {
    "luxury": {
        "contexts": ["luxury beach resort", "5-star city hotel", "luxury spa retreat", "premium mountain resort"],
        "base_price_range": (300, 800),
        "amenities": ["spa", "concierge", "fine dining", "butler service", "private beach", "helicopter pad"],
        "room_types": ["Suite", "Penthouse", "Villa", "Presidential Suite"]
    },
    "business": {
        "contexts": ["city center business hotel", "airport hotel", "convention center hotel", "downtown executive hotel"],
        "base_price_range": (120, 300),
        "amenities": ["business center", "conference rooms", "wifi", "airport shuttle", "meeting facilities", "express check-in"],
        "room_types": ["Standard Room", "Executive Room", "Business Suite", "Deluxe Room"]
    },
    "leisure": {
        "contexts": ["family resort", "beach hotel", "mountain lodge", "countryside inn"],
        "base_price_range": (80, 250),
        "amenities": ["pool", "restaurant", "bar", "playground", "sports facilities", "entertainment"],
        "room_types": ["Standard Room", "Family Room", "Superior Room", "Deluxe Room"]
    },
    "boutique": {
        "contexts": ["boutique hotel", "historic hotel", "design hotel", "art hotel"],
        "base_price_range": (150, 400),
        "amenities": ["unique decor", "local cuisine", "art gallery", "rooftop bar", "designer rooms"],
        "room_types": ["Deluxe Room", "Superior Room", "Designer Suite", "Artist Room"]
    },
    "budget": {
        "contexts": ["budget hotel", "economy hotel", "hostel", "inn"],
        "base_price_range": (40, 120),
        "amenities": ["wifi", "basic breakfast", "shared facilities", "luggage storage"],
        "room_types": ["Standard Room", "Economy Room", "Shared Room", "Basic Room"]
    }
}

# Customer personas
CUSTOMER_PERSONAS = {
    "business_traveler": {
        "priorities": ["location", "wifi", "quick service", "airport access"],
        "language_style": "efficient, professional",
        "typical_questions": ["airport shuttle", "business center", "early check-in", "express checkout"],
        "price_sensitivity": "medium",
        "booking_pattern": "quick decisions, specific dates"
    },
    "family_vacation": {
        "priorities": ["family rooms", "amenities for kids", "value for money", "safety"],
        "language_style": "friendly, detailed questions",
        "typical_questions": ["family rooms", "kids activities", "pool safety", "nearby attractions"],
        "price_sensitivity": "high",
        "booking_pattern": "comparison shopping, advance planning"
    },
    "luxury_seeker": {
        "priorities": ["premium amenities", "service quality", "exclusive experiences"],
        "language_style": "sophisticated, expects high standards",
        "typical_questions": ["suite options", "spa services", "fine dining", "concierge"],
        "price_sensitivity": "low",
        "booking_pattern": "quality over price, detailed service inquiries"
    },
    "budget_conscious": {
        "priorities": ["price", "basic amenities", "location"],
        "language_style": "direct, price-focused",
        "typical_questions": ["cheapest option", "what's included", "free amenities", "discounts"],
        "price_sensitivity": "very high",
        "booking_pattern": "extensive comparison, looking for deals"
    },
    "solo_traveler": {
        "priorities": ["safety", "location", "social opportunities"],
        "language_style": "cautious but adventurous",
        "typical_questions": ["safe area", "single occupancy", "local recommendations", "transport"],
        "price_sensitivity": "medium",
        "booking_pattern": "research-heavy, flexible dates"
    },
    "romantic_getaway": {
        "priorities": ["privacy", "ambiance", "special services"],
        "language_style": "emotional, seeking experiences",
        "typical_questions": ["romantic packages", "private dining", "spa for couples", "special occasions"],
        "price_sensitivity": "low",
        "booking_pattern": "experience-focused, willing to pay premium"
    }
}

MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

# Seasonal pricing modifiers
SEASONAL_MODIFIERS = {
    "peak": 1.4,
    "shoulder": 1.1,
    "off": 0.8 
}