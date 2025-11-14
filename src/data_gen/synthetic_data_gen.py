"""
Main script for generating synthetic hotel chatbot conversations
"""
import json
import logging
import random
from pathlib import Path
from typing import List, Dict
import argparse

from config import ModelConfig, GenerationConfig
from model_utils import ModelHandler
from conversation_utils import ConversationPrompts, parse_generated_conversation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConversationGenerator:
    """Main class for generating synthetic conversations"""
    
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.model_config = model_config
        self.generation_config = generation_config
        self.model_handler = ModelHandler(model_config)
        self.prompt_generator = ConversationPrompts()
        
    def generate_conversations(self) -> List[Dict]:
        """Generate all conversations according to configuration"""
        logger.info("Starting conversation generation...")
        
        # Load model
        self.model_handler.load_model()
        
        all_conversations = []
        
        # Generate for each intent type
        intent_methods = {
            "low": self.prompt_generator.generate_low_intent_prompt,
            "medium": self.prompt_generator.generate_medium_intent_prompt,
            "high": self.prompt_generator.generate_high_intent_prompt,
            "abandoned": self.prompt_generator.generate_abandoned_intent_prompt
        }
        
        for intent, method in intent_methods.items():
            target_count = int(
                self.generation_config.conversations_per_intent * 
                self.generation_config.intent_weights[intent]
            )
            logger.info(f"Generating {target_count} conversations for '{intent}' intent")
            
            intent_conversations = self._generate_for_intent(intent, method, target_count)
            all_conversations.extend(intent_conversations)
            
            logger.info(f"Generated {len(intent_conversations)} conversations for '{intent}' intent")
        
        # Clean up model
        self.model_handler.cleanup()
        
        logger.info(f"Total conversations generated: {len(all_conversations)}")
        return all_conversations
    
    def _generate_for_intent(self, intent: str, prompt_method, target_count: int) -> List[Dict]:
        """Generate conversations for a specific intent"""
        conversations = []
        retries = 0
        max_total_retries = target_count * self.generation_config.max_retries
        
        while len(conversations) < target_count and retries < max_total_retries:
            try:
                # Generate prompt
                prompt = prompt_method()
                
                # Generate conversation
                generated_text = self.model_handler.generate_text(prompt)
                
                # Parse and validate
                conversation = parse_generated_conversation(generated_text)
                
                # Verify intent matches
                if conversation.get("intent_label") == intent:
                    conversations.append(conversation)
                    logger.debug(f"Successfully generated conversation {len(conversations)}/{target_count} for {intent}")
                else:
                    logger.warning(f"Generated conversation has wrong intent: {conversation.get('intent_label')} != {intent}")
                    retries += 1
                    
            except Exception as e:
                logger.warning(f"Failed to generate conversation for {intent}: {e}")
                retries += 1
                continue
        
        if len(conversations) < target_count:
            logger.warning(f"Could only generate {len(conversations)}/{target_count} conversations for {intent}")
        
        return conversations
    
    def save_conversations(self, conversations: List[Dict], output_path: Path):
        """Save conversations to JSONL file"""
        logger.info(f"Saving {len(conversations)} conversations to {output_path}")
        
        # Shuffle conversations
        random.shuffle(conversations)
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        logger.info(f"Conversations saved successfully!")
    
    def generate_statistics(self, conversations: List[Dict]) -> Dict:
        """Generate statistics about the dataset"""
        stats = {
            "total_conversations": len(conversations),
            "intent_distribution": {},
            "avg_messages_per_conversation": 0,
            "completed_bookings": 0,
            "abandoned_bookings": 0
        }
        
        total_messages = 0
        
        for conv in conversations:
            intent = conv.get("intent_label", "unknown")
            stats["intent_distribution"][intent] = stats["intent_distribution"].get(intent, 0) + 1
            total_messages += len(conv.get("messages", []))
            
            if conv.get("booking_completed"):
                stats["completed_bookings"] += 1
            if intent == "abandoned":
                stats["abandoned_bookings"] += 1
        
        stats["avg_messages_per_conversation"] = total_messages / len(conversations) if conversations else 0
        
        return stats

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate synthetic hotel chatbot conversations")
    parser.add_argument("--output", "-o", type=str, default="synthetic_conversations.jsonl",
                       help="Output file path")
    parser.add_argument("--count", "-c", type=int, default=200,
                       help="Total number of conversations to generate")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name to use")
    
    args = parser.parse_args()
    
    # Setup configurations
    model_config = ModelConfig(model_name=args.model)
    generation_config = GenerationConfig(conversations_per_intent=args.count // 4)
    
    # Generate conversations
    generator = ConversationGenerator(model_config, generation_config)
    conversations = generator.generate_conversations()
    
    # Save results
    output_path = Path(args.output)
    generator.save_conversations(conversations, output_path)
    
    # Generate and print statistics
    stats = generator.generate_statistics(conversations)
    print("\n=== Dataset Statistics ===")
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Average messages per conversation: {stats['avg_messages_per_conversation']:.1f}")
    print(f"Intent distribution: {stats['intent_distribution']}")
    print(f"Completed bookings: {stats['completed_bookings']}")
    print(f"Abandoned bookings: {stats['abandoned_bookings']}")

if __name__ == "__main__":
    main()