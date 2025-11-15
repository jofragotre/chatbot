"""
Main script for generating synthetic hotel chatbot conversations with batch processing
"""
import json
import logging
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

from config import ModelConfig, GenerationConfig
from model_utils import ModelHandler
from conversation_utils import ConversationPrompts, parse_generated_conversation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchedConversationGenerator:
    """Main class for generating synthetic conversations with batch processing"""
    
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.model_config = model_config
        self.generation_config = generation_config
        self.model_handler = ModelHandler(model_config)
        self.prompt_generator = ConversationPrompts()
        self.batch_size = generation_config.batch_size
        
    def generate_conversations(self) -> List[Dict]:
        """Generate all conversations according to configuration"""
        logger.info("Starting batched conversation generation...")
        
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
                self.generation_config.conversations_per_intent
            )
            logger.info(f"Generating {target_count} conversations for '{intent}' intent")
            
            intent_conversations = self._generate_for_intent_batched(intent, method, target_count)
            all_conversations.extend(intent_conversations)
            
            logger.info(f"Generated {len(intent_conversations)} conversations for '{intent}' intent")
        
        # Clean up model
        self.model_handler.cleanup()
        
        logger.info(f"Total conversations generated: {len(all_conversations)}")
        return all_conversations
    
    def _generate_for_intent_batched(self, intent: str, prompt_method, target_count: int) -> List[Dict]:
        """Generate conversations for a specific intent using batch processing"""
        conversations = []
        
        # Calculate number of batches needed
        total_batches = math.ceil(target_count / self.batch_size)
        
        with tqdm(total=target_count, desc=f"Generating {intent} conversations") as pbar:
            batch_num = 0
            
            while len(conversations) < target_count:

                # Calculate current batch size
                remaining = target_count - len(conversations)
                current_batch_size = min(self.batch_size, remaining)
                
                # Generate batch of prompts
                prompts = [prompt_method() for _ in range(current_batch_size)]
                
                # Generate batch of conversations
                batch_results = self._generate_conversation_batch(prompts, intent)
                
                # Add successful conversations
                successful_convs = [conv for conv in batch_results if conv is not None]
                conversations.extend(successful_convs)
                
                # Update progress
                pbar.update(len(successful_convs))
                
                # Log batch results
                success_rate = len(successful_convs) / len(prompts) * 100
                logger.debug(f"Batch {batch_num + 1}: {len(successful_convs)}/{len(prompts)} "
                            f"successful ({success_rate:.1f}%)")
                
                batch_num += 1
        
        if len(conversations) < target_count:
            logger.warning(f"Could only generate {len(conversations)}/{target_count} conversations for {intent}")
        
        return conversations
    
    def _generate_conversation_batch(self, prompts: List[str], expected_intent: str) -> List[Dict]:
        """Generate and parse a batch of conversations"""
        # Generate batch
        generated_texts = self.model_handler.generate_text_batch(prompts)
        
        # Parse each generated conversation
        parsed_conversations = []
        for i, generated_text in enumerate(generated_texts):
            try:
                conversation = parse_generated_conversation(generated_text)
                # Verify intent matches
                if conversation.get("intent_label") == expected_intent:
                    parsed_conversations.append(conversation)
                else:
                    logger.debug(f"Batch item {i}: Wrong intent - expected {expected_intent}, "
                               f"got {conversation.get('intent_label')}")
                    parsed_conversations.append(None)
                    
            except Exception as e:
                logger.debug(f"Batch item {i}: Parse failed - {e}")
                parsed_conversations.append(None)
        
        return parsed_conversations
    
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
    parser = argparse.ArgumentParser(description="Generate synthetic hotel chatbot conversations with batching")
    parser.add_argument("--output", "-o", type=str, default="synthetic_conversations.jsonl",
                       help="Output file path")
    parser.add_argument("--count", "-c", type=int, default=10000,
                       help="Total number of conversations to generate")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-1.7B",
                       help="Model name to use")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                       help="Batch size for generation")
    
    args = parser.parse_args()
    
    # Setup configurations
    model_config = ModelConfig(model_name=args.model)
    generation_config = GenerationConfig(
        conversations_per_intent=args.count // 4,
        batch_size=args.batch_size
    )
    
    # Generate conversations
    generator = BatchedConversationGenerator(model_config, generation_config)
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