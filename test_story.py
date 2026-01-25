import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import os

# 1. SETUP: Load the model you just trained
# We load from the local folder where story_generator.py saved it
model_path = "./gpt2-story-finetuned" 

print(f"Loading model from {model_path}...")

if not os.path.exists(model_path):
    print(f"Error: The folder '{model_path}' does not exist.")
    print("Please run 'story_generator.py' first to train and save the model.")
    exit()

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. DEVICE: Move to RTX 4060 (GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on: {device}")

def generate_story_segment(prompt, max_new_tokens=100):
    """Generates a continuation of the story based on the prompt."""
    
    # CRITICAL FIX: We use tokenizer() instead of tokenizer.encode()
    # This returns a dictionary containing both 'input_ids' and 'attention_mask'
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,           # Pass the token IDs
        attention_mask=inputs.attention_mask, # Pass the mask (Fixes the warning/error)
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1, 
        temperature=0.8,
        top_k=50,             
        top_p=0.95,           
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode back to text
    segment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return segment

def generate_choices():
    """Simple placeholder for user choices."""
    options = [
        f"Choice 1: Explore the {random.choice(['dark cave', 'shimmering lake', 'ruined tower'])}.",
        f"Choice 2: Talk to the {random.choice(['mysterious stranger', 'nervous merchant', 'wandering knight'])}.",
        f"Choice 3: Check your inventory for {random.choice(['weapons', 'potions', 'maps'])}."
    ]
    return options

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    initial_prompt = "Once upon a time, in a kingdom made of glass,"
    
    print("\n" + "="*40)
    print("INPUT PROMPT:", initial_prompt)
    print("="*40)
    
    # Generate the story
    story_output = generate_story_segment(initial_prompt)
    
    print("\nAI STORY OUTPUT:")
    print(story_output)
    
    print("\nPLAYER CHOICES:")
    for choice in generate_choices():
        print(choice)
    print("="*40)