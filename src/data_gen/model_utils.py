"""
Model loading and inference utilities
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles model loading and text generation"""
    
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
    
    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare messages for chat template
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
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def cleanup(self):
        """Clean up model and free memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()