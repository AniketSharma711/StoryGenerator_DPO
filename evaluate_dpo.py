import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
SFT_MODEL_PATH = "./gpt2-story-finetuned"   # The old model
DPO_MODEL_PATH = "./dpo_results"            # The new aligned model
PROMPTS = [
    "Once upon a time, in a dark forest",
    "A brave knight named Arthur",
    "The little robot wanted to fly",
    "One sunny morning, a cat"
]

def generate_story(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with some creativity settings
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,      # Generate ~100 words
        do_sample=True,          # Add randomness/creativity
        temperature=0.7,         # Control creativity (0.7 is balanced)
        top_k=50,                # Limit crazy words
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Running on {device}...")

    # 1. Load Tokenizer (Same for both)
    print("⏳ Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Load OLD Model (SFT)
    print("⏳ Loading SFT Model (Old)...")
    model_sft = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(device)

    # 3. Load NEW Model (DPO)
    print("⏳ Loading DPO Model (New)...")
    model_dpo = AutoModelForCausalLM.from_pretrained(DPO_MODEL_PATH).to(device)

    print("\n" + "="*60)
    print("⚔️  MODEL BATTLE: SFT vs. DPO  ⚔️")
    print("="*60 + "\n")

    for prompt in PROMPTS:
        # Set seed so they get the same "luck"
        torch.manual_seed(42)
        story_sft = generate_story(model_sft, tokenizer, prompt)
        
        torch.manual_seed(42) # Reset seed for fair comparison
        story_dpo = generate_story(model_dpo, tokenizer, prompt)

        print(f"📝 PROMPT: {prompt}")
        print("-" * 20)
        print(f"🔴 SFT (Old): {story_sft}")
        print("-" * 20)
        print(f"🟢 DPO (New): {story_dpo}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()