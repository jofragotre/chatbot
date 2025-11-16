"""
Conversation generation prompts and utilities with enhanced context and few-shot examples
"""
import json
import random
from typing import Dict, List, Optional, Tuple
from config import HOTEL_CATEGORIES, CUSTOMER_PERSONAS, MONTHS, SEASONAL_MODIFIERS

class ConversationPrompts:
    """Generate prompts for different booking intent levels with enhanced context"""
    
    # Few-shot examples from real data
    FEW_SHOT_EXAMPLES = {
        "low": [
            {
                "conversation": [
                    {"role": "user", "text": "Do you have an airport shuttle?"},
                    {"role": "bot", "text": "Yes, the shuttle runs every hour from 6:00 to 22:00. It costs €12 per person."},
                    {"role": "user", "text": "Where do I catch it at the airport?"},
                    {"role": "bot", "text": "Terminal 1, exit B, bay 5. Look for the hotel logo on the sign."}
                ],
                "intent": "low"
            },
            {
                "conversation": [
                    {"role": "user", "text": "Any good seafood places within walking distance?"},
                    {"role": "bot", "text": "Try Mar Azul (10 min) for grilled fish and O Porto Velho (8 min) for seafood rice."},
                    {"role": "user", "text": "Do I need a reservation?"},
                    {"role": "bot", "text": "Recommended after 19:30. I can share phone numbers if you like."}
                ],
                "intent": "low"
            }
        ],
        "medium": [
            {
                "conversation": [
                    {"role": "user", "text": "What's your early check-in policy?"},
                    {"role": "bot", "text": "Standard check-in is 15:00. Early check-in from 12:00 is subject to room readiness."},
                    {"role": "user", "text": "Is there a fee?"},
                    {"role": "bot", "text": "There's no fee if the room is ready. Otherwise, we can store luggage for free."}
                ],
                "intent": "medium"
            },
            {
                "conversation": [
                    {"role": "user", "text": "Do you have sea-view rooms this weekend?"},
                    {"role": "bot", "text": "Yes: Standard €150, Superior €190 with balcony."},
                    {"role": "user", "text": "What's the difference?"},
                    {"role": "bot", "text": "Superior has a balcony and premium breakfast."}
                ],
                "intent": "medium"
            }
        ],
        "high": [
            {
                "conversation": [
                    {"role": "user", "text": "Hi, availability for 12–14 Oct for 2 adults?"},
                    {"role": "bot", "text": "Available: Deluxe €180/night. Proceed to book?"},
                    {"role": "user", "text": "Yes, book the Deluxe."},
                    {"role": "bot", "text": "Name and email, please."},
                    {"role": "user", "text": "John Smith, john@example.com"},
                    {"role": "bot", "text": "Payment processed. Booking confirmed."}
                ],
                "intent": "high"
            }
        ],
        "abandoned": [
            {
                "conversation": [
                    {"role": "user", "text": "I need 1 night tonight."},
                    {"role": "bot", "text": "Single Room €120. Proceed?"},
                    {"role": "user", "text": "Yes."},
                    {"role": "bot", "text": "Enter card details to guarantee."},
                    {"role": "user", "text": "I'll finish later."},
                    {"role": "bot", "text": "I can hold the rate for 15 minutes."}
                ],
                "intent": "abandoned"
            }
        ]
    }
    
    @staticmethod
    def get_enhanced_context(use_persona: bool = True) -> Dict[str, str]:
        """Get enhanced hotel context with persona and seasonal considerations"""
        
        # Select hotel category
        category = random.choice(list(HOTEL_CATEGORIES.keys()))
        hotel_data = HOTEL_CATEGORIES[category]
        
        # Select persona if requested
        persona = None
        if use_persona:
            persona = random.choice(list(CUSTOMER_PERSONAS.keys()))
            persona_data = CUSTOMER_PERSONAS[persona]
        
        # Generate realistic dates
        month = random.choice(MONTHS)
        season = ConversationPrompts._get_season(month)
        day_interval = random.randint(1, 8)
        first_day = random.randint(1, 28 - day_interval) # Assume 28 days per month and single month stay for simplicity
        last_day = first_day + day_interval
        
        # Calculate realistic pricing
        base_price = random.randint(*hotel_data["base_price_range"])
        seasonal_price = int(base_price * SEASONAL_MODIFIERS[season])
        
        # Generated context
        context = {
            "hotel_category": category,
            "hotel_type": random.choice(hotel_data["contexts"]),
            "amenities": random.sample(hotel_data["amenities"], k=random.randint(3, min(6, len(hotel_data["amenities"])))),
            "room_types": random.sample(hotel_data["room_types"], k=random.randint(2, len(hotel_data["room_types"]))),
            "base_price": base_price,
            "seasonal_price": seasonal_price,
            "price_range": f"€{seasonal_price}-{seasonal_price + 100}",
            "date_range": f"{month} {first_day} - {last_day}",
            "season": season,
            "persona": persona,
            "persona_data": CUSTOMER_PERSONAS.get(persona, {}) if persona else {}
        }
        
        return context
    
    @staticmethod
    def _get_season(month: str) -> str:
        """Determine season based on month"""
        if month in ["June", "July", "August", "December"]:
            return "peak"
        elif month in ["March", "April", "May", "September", "October"]:
            return "shoulder"
        else:
            return "off"
    
    @staticmethod
    def _get_few_shot_examples(intent: str, include_examples: bool) -> str:
        """Generate few-shot examples for the prompt"""
        if not include_examples or intent not in ConversationPrompts.FEW_SHOT_EXAMPLES:
            return ""
        
        examples = ConversationPrompts.FEW_SHOT_EXAMPLES[intent]
        example_text = "Here are examples of similar conversations:\n\n"
        
        for i, example in enumerate(examples[:2]):  # Limit to 2 examples
            example_text += f"Example {i+1}:\n"
            for msg in example["conversation"]:
                example_text += f"{msg['role'].upper()}: {msg['text']}\n"
            example_text += f"Intent: {example['intent']}\n\n"
        
        example_text += "Now generate a similar but different conversation:\n\n"
        return example_text
    
    @staticmethod
    def generate_low_intent_prompt(use_persona: bool = True, include_examples: bool = True) -> str:
        """Generate enhanced prompt for Low/Exploring intent conversations"""
        context = ConversationPrompts.get_enhanced_context(use_persona)
        
        few_shot = ConversationPrompts._get_few_shot_examples("low", include_examples)
        
        persona_context = ""
        if context["persona"]:
            persona_data = context["persona_data"]
            persona_context = f"Customer persona: {context['persona'].replace('_', ' ')}\n" \
                            f"Typical behavior: {persona_data.get('language_style', '')}\n" \
                            f"Likely questions: {', '.join(persona_data.get('typical_questions', [])[:3])}\n\n"
        
        prompt = (
            f"{few_shot}"
            f"Generate a realistic hotel chatbot conversation showing LOW booking intent (Exploring phase).\n\n"
            
            f"Hotel Context:\n"
            f"- Type: {context['hotel_type']}\n"
            f"- Category: {context['hotel_category']}\n" 
            f"- Amenities: {', '.join(context['amenities'])}\n"
            f"- Season: {context['season']}\n\n"
            
            f"{persona_context}"
            
            "The conversation should show:\n"
            "- Broad, general questions about the location/hotel/services\n"
            "- Casual browsing behavior without specific booking intent\n"
            "- General questions: could be about area, amenities, general policies\n"
            "- No specific dates, room types, or pricing inquiries\n"
            "- 4-7 message exchanges\n"
            "- Natural, conversational flow\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."}, {"role": "bot", "text": "..."}], "intent_label": "low"}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt
    
    @staticmethod
    def generate_medium_intent_prompt(use_persona: bool = True, include_examples: bool = True) -> str:
        """Generate enhanced prompt for Medium/Evaluating intent conversations"""
        context = ConversationPrompts.get_enhanced_context(use_persona)
        
        few_shot = ConversationPrompts._get_few_shot_examples("medium", include_examples)
        
        persona_context = ""
        if context["persona"]:
            persona_data = context["persona_data"]
            persona_context = f"Customer persona: {context['persona'].replace('_', ' ')}\n" \
                            f"Price sensitivity: {persona_data.get('price_sensitivity', 'medium')}\n" \
                            f"Decision style: {persona_data.get('booking_pattern', '')}\n\n"
        
        prompt = (
            f"{few_shot}"
            f"Generate a realistic hotel chatbot conversation showing MEDIUM booking intent (Evaluating phase).\n\n"
            
            f"Hotel Context:\n"
            f"- Type: {context['hotel_type']}\n"
            f"- Available rooms: {', '.join(context['room_types'])}\n"
            f"- Price range: {context['price_range']}\n"
            f"- Sample dates: {context['date_range']}\n\n"
            
            f"{persona_context}"
            
            "The conversation should show:\n"
            "- Specific questions about rates, availability, room types, or policies\n"
            "- Comparison shopping behavior\n"
            "- Questions about cancellation policies, what's included, terms\n"
            "- Shows genuine interest but no immediate commitment\n"
            "- May ask for specific dates or room comparisons\n"
            "- 5-10 message exchanges\n"
            "- Natural decision-making process\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."}, {"role": "bot", "text": "..."}], "intent_label": "medium"}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt
    
    @staticmethod
    def generate_high_intent_prompt(use_persona: bool = True, include_examples: bool = True) -> str:
        """Generate enhanced prompt for High/Actioning intent conversations"""
        context = ConversationPrompts.get_enhanced_context(use_persona)
        
        few_shot = ConversationPrompts._get_few_shot_examples("high", include_examples)
        
        persona_context = ""
        if context["persona"]:
            persona_context = f"Customer persona: {context['persona'].replace('_', ' ')}\n" \
                            f"Booking style: ready to commit, decisive\n\n"
        
        prompt = (
            f"{few_shot}"
            f"Generate a realistic hotel chatbot conversation showing HIGH booking intent (Actioning phase).\n\n"
            
            f"Hotel Context:\n"
            f"- Type: {context['hotel_type']}\n"
            f"- Available rooms: {', '.join(context['room_types'][:2])}\n"
            f"- Price: around €{context['seasonal_price']}/night\n\n"
            
            f"{persona_context}"
            
            "The conversation should show:\n"
            "- Customer ready to book with specific requirements\n"
            "- Direct booking requests: 'Book the [room] for [dates]'\n"
            "- Providing booking details (dates, guest info, contact)\n"
            "- Moving through to payment/confirmation\n"
            "- 5-10 message exchanges\n"
            "- Clear progression from inquiry to booking completion\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."}, {"role": "bot", "text": "..."}], "intent_label": "high", "booking_completed": true}\n\n'
            
            "Generate the conversation:"
        )
        
        return prompt
    
    @staticmethod
    def generate_abandoned_intent_prompt(use_persona: bool = True, include_examples: bool = True) -> str:
        """Generate enhanced prompt for Abandoned intent conversations"""
        context = ConversationPrompts.get_enhanced_context(use_persona)
        
        few_shot = ConversationPrompts._get_few_shot_examples("abandoned", include_examples)
        
        abandonment_reasons = [
            "price concerns", "need to check with partner", "found better deal elsewhere",
            "changed travel plans", "payment issues", "wants to think about it"
        ]
        reason = random.choice(abandonment_reasons)
        
        persona_context = ""
        if context["persona"]:
            persona_context = f"Customer persona: {context['persona'].replace('_', ' ')}\n" \
                            f"Likely abandonment reason: {reason}\n\n"
        
        prompt = (
            f"{few_shot}"
            f"Generate a realistic hotel chatbot conversation showing ABANDONED booking intent.\n\n"
            
            f"Hotel Context:\n"
            f"- Type: {context['hotel_type']}\n"
            f"- Price: {context['price_range']}\n\n"
            
            f"{persona_context}"
            
            "The conversation should show:\n"
            "- Customer progressed toward booking but didn't complete\n"
            "- Got to payment/confirmation step but stopped\n"
            "- Expressions like: 'I'll finish this later', 'Let me think about it', 'Hold the room'\n"
            "- Bot trying to help but customer ultimately not completing\n"
            "- 5-10 message exchanges\n"
            "- Clear progression that stops before completion\n\n"
            
            "Format as JSON with this structure:\n"
            '{"session_id": "random_id", "messages": [{"role": "user", "text": "..."}, {"role": "bot", "text": "..."}], "intent_label": "abandoned", "abandonment_reason": "' + reason + '"}\n\n'
            
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