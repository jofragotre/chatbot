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
    
    # Intent distribution
    intent_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.intent_weights is None:
            self.intent_weights = {
                "low": 0.4,      # 40% exploring conversations
                "medium": 0.3,   # 30% evaluating conversations
                "high": 0.2,     # 20% actioning conversations
                "abandoned": 0.1 # 10% abandoned conversations
            }

# Conversation parameters
HOTEL_CONTEXTS = [
    "luxury beach resort",
    "city center business hotel", 
    "mountain lodge",
    "boutique hotel",
    "family resort",
    "spa hotel",
    "budget hotel",
    "historic hotel",
    "hotel",
    "accomodation"
]

AMENITIES = [
    "spa", "gym", "pool", "restaurant", "bar", "room service", 
    "wifi", "parking", "airport shuttle", "concierge", "laundry",
    "business center", "conference rooms", "pet-friendly", "massage", "fitness center",
    "gymnasium", "sauna", "steam room", "outdoor pool", "indoor pool", "jacuzzi", "all inclusive"
]

ROOM_TYPES = [
    "Standard Room", "Deluxe Room", "Superior Room", "Suite",
    "Family Room", "Executive Room", "Penthouse", "Villa"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
]