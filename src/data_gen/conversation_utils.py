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
        
        return f"""Generate a realistic hotel chatbot conversation showing LOW booking intent (Exploring phase).

Context: {context['hotel_type']} with {', '.join(context['amenities'])}

The conversation should show:
- Broad, general questions about the location/hotel
- Casual browsing behavior
- No specific dates or details
- Questions like "What's nearby?", "Any weekend deals?", "Tell me about the area"
- 3-6 message exchanges
- Natural, conversational flow

Format as JSON with this structure:
{{"session_id": "random_id", "messages": [{{"role": "user", "text": "..."}} or {{"role": "bot", "text": "..."}}], "intent_label": "low"}}

Generate the conversation:"""
    
    @staticmethod
    def generate_medium_intent_prompt() -> str:
        """Generate prompt for Medium/Evaluating intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        return f"""Generate a realistic hotel chatbot conversation showing MEDIUM booking intent (Evaluating phase).

Context: {context['hotel_type']}, rooms: {', '.join(context['room_types'])}, price range: {context['price_range']}

The conversation should show:
- Specific questions about dates, prices, room types
- Comparing options and policies
- Questions like "What's the rate for Oct 15-17?", "What's included in breakfast?", "Cancellation policy?"
- Shows interest but no commitment yet
- 4-8 message exchanges
- Natural decision-making process

Format as JSON with this structure:
{{"session_id": "random_id", "messages": [{{"role": "user", "text": "..."}} or {{"role": "bot", "text": "..."}}], "intent_label": "medium"}}

Generate the conversation:"""
    
    @staticmethod
    def generate_high_intent_prompt() -> str:
        """Generate prompt for High/Actioning intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        return f"""Generate a realistic hotel chatbot conversation showing HIGH booking intent (Actioning phase).

Context: {context['hotel_type']}, available rooms: {', '.join(context['room_types'][:2])}

The conversation should show:
- Ready to book with specific commands
- Providing booking details (dates, guest info)
- Questions like "Book the Deluxe for Oct 12-14", "I'll take that room", "Proceed with booking"
- Booking completion or moving to payment
- 4-7 message exchanges
- Clear intent to finalize booking

Format as JSON with this structure:
{{"session_id": "random_id", "messages": [{{"role": "user", "text": "..."}} or {{"role": "bot", "text": "..."}}], "intent_label": "high", "booking_completed": true/false}}

Generate the conversation:"""
    
    @staticmethod
    def generate_abandoned_intent_prompt() -> str:
        """Generate prompt for Abandoned intent conversations"""
        context = ConversationPrompts.get_base_context()
        
        return f"""Generate a realistic hotel chatbot conversation showing ABANDONED booking intent.

Context: {context['hotel_type']}, price: {context['price_range']}

The conversation should show:
- User was progressing toward booking but didn't complete
- Reached payment/confirmation step but stopped
- Messages like "I'll finish this later", "Let me think about it", "Hold the room for me"
- Session drops off or user postpones
- 5-8 message exchanges
- Shows high intent that wasn't completed

Format as JSON with this structure:
{{"session_id": "random_id", "messages": [{{"role": "user", "text": "..."}} or {{"role": "bot", "text": "..."}}], "intent_label": "abandoned", "abandonment_reason": "reason"}}

Generate the conversation:"""

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