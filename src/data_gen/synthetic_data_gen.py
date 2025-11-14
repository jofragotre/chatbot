import json
import random
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Setup: Load quantized model and tokenizer
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",  # Use M3 GPU if available
    dtype=torch.float16,  # Further memory optimization
    trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Qwen often needs this for generation

# Load existing samples for few shot examples:
json_path = "data/provided/small_sample.jsonl"
msgs = []
with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        msgs.append({"messages": obj["messages"]})

# Categories for balance (e.g., for booking intent signal)
categories = [
    "low intent: broad questions like 'What's nearby?'",
    "medium intent: specifics like dates/prices",
    "high intent: direct booking with completion",
    "abandoned high intent: starts booking but drops off"
]

def generate_conversation(category):
    # Build messages for Qwen chat template
    messages = [
        {"role": "system", "content": "You are a simulator for hotel chatbot conversations. "
         "Generate realistic, varied sessions in valid JSON format only. "
         "Match this schema: {'messages': list of dicts with 'role' (user/bot) and 'text', "
         "optional 'booking': 'completed' or 'abandoned'}. "
         "Avoid biases; make diverse (e.g., different names, topics). "
         "Few-shot examples:"
         f"{msgs}"
         f"Now generate a NEW unique session for: {category}. Vary length (3-10 turns), include realistic bot responses. Output ONLY the JSON."}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,  # Limit length
        temperature=0.7,  # Creativityç
        top_p=0.9,  # Diversity
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract and clean JSON (LLM might add extras)
    try:
        # Find the JSON part after the prompt
        json_start = generated_text.find("{")
        if json_start == -1:
            raise ValueError("No JSON found")
        json_str = generated_text[json_start:]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        print(f"Invalid JSON for {category}: {json_str}, retrying...")
        return json_str

# Generate N sessions, balanced across categories
num_sessions = 100
synthetic_data = []
for i in tqdm.tqdm(range(num_sessions)):
    category = random.choice(categories)
    session = generate_conversation(category)
    # session["session_id"] = f"synthetic_{i:04d}"  # Unique ID
    synthetic_data.append(session)
    print(f"Generated session {i+1}/{num_sessions} for {category}")

# Save to file
with open("synthetic_data.txt", "w") as f:
    for line in synthetic_data:
        f.write(f"{line}\n")

print("Synthetic data generated and saved to synthetic_data.json!")