"""
Conversation generation prompts and utilities
"""
import json
import random
from typing import Dict, List
from config import HOTEL_CONTEXTS, AMENITIES, ROOM_TYPES

class ConversationPrompts:
    """Generate prompts for different booking intent levels"""
    
    @staticmethod
    def get_base_context() -> Dict[str, str]:
        """Get randomized hotel context"""
        return {
            "hotel_type": random.choice(HOTEL_CONTEXTS),
            "amenities": random.sample(AMENITIES, k=random.randint(3, 6)),
            "room_types": random.sample(ROOM_TYPES, k=random.randint(2, 4)),
            "price_range": random.choice(["€80-120", "€150-200", "€300-500", "€500+"]),
        }
    
    @staticmethod
    def generate_low_intent_prompt() -> str:
        """Generate prompt for Low/Exploring intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        prompt = (
            "Generate a realistic hotel chatbot conversation showing LOW booking intent (Exploring phase).\n\n"
            
            f"Context: {context['hotel_type']} with {', '.join(context['amenities'])}\n\n"
            
            "The conversation should show:\n"
            "- Broad, general questions about the location/hotel\n"
            "- Casual browsing behavior\n"
            "- No specific dates or details\n"
            "- Questions like 'What's nearby?', 'Any weekend deals?', 'Tell me about the area'\n"
            "- 3-6 message exchanges\n"
            "- Natural, conversational flow\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."} or {"role": "bot", "text": "..."}], "intent_label": "low"}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt
    
    @staticmethod
    def generate_medium_intent_prompt() -> str:
        """Generate prompt for Medium/Evaluating intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        prompt = (
            "Generate a realistic hotel chatbot conversation showing MEDIUM booking intent (Evaluating phase).\n\n"
            
            f"Context: {context['hotel_type']}, "
            f"rooms: {', '.join(context['room_types'])}, "
            f"price range: {context['price_range']}\n\n"
            
            "The conversation should show:\n"
            "- Specific questions about dates, prices, room types\n"
            "- Comparing options and policies\n"
            "- Questions like 'What's the rate for Oct 15-17?', 'What's included in breakfast?', 'Cancellation policy?'\n"
            "- Shows interest but no commitment yet\n"
            "- 4-8 message exchanges\n"
            "- Natural decision-making process\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."} or {"role": "bot", "text": "..."}], "intent_label": "medium"}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt
    
    @staticmethod
    def generate_high_intent_prompt() -> str:
        """Generate prompt for High/Actioning intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        prompt = (
            "Generate a realistic hotel chatbot conversation showing HIGH booking intent (Actioning phase).\n\n"
            
            f"Context: {context['hotel_type']}, "
            f"available rooms: {', '.join(context['room_types'][:2])}\n\n"
            
            "The conversation should show:\n"
            "- Ready to book with specific commands\n"
            "- Providing booking details (dates, guest info)\n"
            "- Questions like 'Book the Deluxe for Oct 12-14', 'I'll take that room', 'Proceed with booking'\n"
            "- Booking completion or moving to payment\n"
            "- 4-7 message exchanges\n"
            "- Clear intent to finalize booking\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."} or {"role": "bot", "text": "..."}], "intent_label": "high", "booking_completed": true/false}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt
    
    @staticmethod
    def generate_abandoned_intent_prompt() -> str:
        """Generate prompt for Abandoned intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        prompt = (
            "Generate a realistic hotel chatbot conversation showing ABANDONED booking intent.\n\n"
            
            f"Context: {context['hotel_type']}, "
            f"price: {context['price_range']}\n\n"
            
            "The conversation should show:\n"
            "- User was progressing toward booking but didn't complete\n"
            "- Reached payment/confirmation step but stopped\n"
            "- Messages like 'I'll finish this later', 'Let me think about it', 'Hold the room for me'\n"
            "- Session drops off or user postpones\n"
            "- 5-8 message exchanges\n"
            "- Shows high intent that wasn't completed\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."} or {"role": "bot", "text": "..."}], "intent_label": "abandoned", "abandonment_reason": "reason"}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt

def parse_generated_conversation(generated_text: str) -> Dict:
    """Parse and validate generated conversation"""
    try:
        # Extract JSON from the generated text
        start_idx = generated_text.find('{')
        end_idx = generated_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in generated text")
        
        json_str = generated_text[start_idx:end_idx]
        conversation = json.loads(json_str)
        
        # Validate structure
        required_fields = ["session_id", "messages", "intent_label"]
        for field in required_fields:
            if field not in conversation:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate messages structure
        for msg in conversation["messages"]:
            if "role" not in msg or "text" not in msg:
                raise ValueError("Invalid message structure")
            if msg["role"] not in ["user", "bot"]:
                raise ValueError(f"Invalid role: {msg['role']}")
        
        return conversation
        
    except Exception as e:
        raise ValueError(f"Failed to parse conversation: {e}")

def generate_session_id() -> str:
    """Generate a random session ID"""
    import uuid
    return str(uuid.uuid4())[:8]

# Additional utility functions for enhanced prompt generation
class PromptTemplates:
    """Template strings for consistent prompt formatting"""
    
    BASE_INSTRUCTIONS = (
        "Generate a realistic hotel chatbot conversation.\n"
        "Use natural language and realistic booking scenarios.\n"
        "Include both user and bot messages in proper sequence.\n\n"
    )
    
    JSON_FORMAT_INSTRUCTION = (
        "Format as JSON with this exact structure:\n"
        '{\n'
        '    "session_id": "random_id",\n'
        '    "messages": [\n'
        '        {"role": "user", "text": "user message"},\n'
        '        {"role": "bot", "text": "bot response"}\n'
        '    ],\n'
        '    "intent_label": "intent_type"\n'
        '}\n\n'
    )
    
    @staticmethod
    def get_context_description(context: Dict[str, str]) -> str:
        """Format context into readable description"""
        return (
            f"Hotel Type: {context['hotel_type']}\n"
            f"Available Amenities: {', '.join(context['amenities'])}\n"
            f"Room Types: {', '.join(context['room_types'])}\n"
            f"Price Range: {context['price_range']}\n"
        )
    
    @staticmethod
    def get_intent_guidelines(intent_type: str) -> str:
        """Get specific guidelines for each intent type"""
        guidelines = {
            "low": [
                "Broad, exploratory questions",
                "No specific dates or booking details",
                "General interest in location/amenities",
                "Questions about deals or promotions",
                "Casual, browsing tone"
            ],
            
            "medium": [
                "Specific questions about availability and pricing",
                "Comparing different room types or dates",
                "Asking about policies (cancellation, breakfast, etc.)",
                "Shows genuine interest but still deciding",
                "More detailed inquiries"
            ],
            
            "high": [
                "Ready to book with clear intent",
                "Providing specific dates and requirements",
                "Asking for booking confirmation",
                "Moving toward payment process",
                "Decisive language and commitment"
            ],
            
            "abandoned": [
                "Started booking process but didn't complete",
                "Asked to hold reservation or finish later",
                "Showed high intent but session ended abruptly",
                "Reached payment step but backed out",
                "Clear signs of abandonment"
            ]
        }
        
        items = guidelines.get(intent_type, [])
        return "\n".join(f"- {item}" for item in items)

# Enhanced conversation examples for better generation
class ConversationExamples:
    """Provide examples to guide better generation"""
    
    LOW_INTENT_EXAMPLE = {
        "session_id": "example_low",
        "messages": [
            {"role": "user", "text": "Hi, what's around your hotel?"},
            {"role": "bot", "text": "We're located in the city center with shopping and restaurants within walking distance."},
            {"role": "user", "text": "Any good weekend packages?"},
            {"role": "bot", "text": "We have a weekend getaway package starting at €199 including breakfast."}
        ],
        "intent_label": "low"
    }
    
    HIGH_INTENT_EXAMPLE = {
        "session_id": "example_high", 
        "messages": [
            {"role": "user", "text": "I want to book a room for Oct 15-17, 2 adults"},
            {"role": "bot", "text": "Available: Deluxe Room €180/night. Shall I proceed?"},
            {"role": "user", "text": "Yes, book it please"},
            {"role": "bot", "text": "I'll need your name and email to confirm."}
        ],
        "intent_label": "high",
        "booking_completed": False
    }