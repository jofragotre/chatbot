"""
Model loading and inference utilities with batch support
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles model loading and batch text generation"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=getattr(torch, self.config.torch_dtype) if self.config.torch_dtype != "auto" else "auto",
                device_map=self.config.device_map,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_text_batch(self, prompts: List[str]) -> List[str]:
        """Generate text from a batch of prompts"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not prompts:
            return []
        
        logger.debug(f"Processing batch of {len(prompts)} prompts")
        
        # Prepare all messages for chat template
        all_messages = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant specialized in generating realistic hotel chatbot conversations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            all_messages.append(messages)
        
        # Apply chat template to all prompts
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        
        # Tokenize batch with padding
        model_inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            padding_side="left",
            max_length=2048
        ).to(self.model.device)
        
        # Generate batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
        
        # Decode only the new tokens for each item in batch
        responses = []
        for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # Extract only the newly generated tokens
            new_tokens = output_ids[len(input_ids):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def generate_text(self, prompt: str) -> str:
        """Generate text from a single prompt (wrapper for batch method)"""
        responses = self.generate_text_batch([prompt])
        return responses[0] if responses else ""
    
    def cleanup(self):
        """Clean up model and free memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()