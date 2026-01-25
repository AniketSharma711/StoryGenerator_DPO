import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# --- CONFIGURATION ---
MODEL_PATH = "./gpt2-story-finetuned"  # Your SFT model (Policy & Reference)
DATA_FILE = "rm_dataset.csv"           # Your dataset
OUTPUT_DIR = "./dpo_results"

def load_and_format_dataset(file_path):
    """
    Loads the CSV and ensures it has the columns DPO needs:
    ['prompt', 'chosen', 'rejected']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. You need preference pairs first!")
    
    df = pd.read_csv(file_path)
    
    # CHECK: Do columns match?
    required_cols = {'prompt', 'chosen', 'rejected'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required_cols}. Found: {df.columns}")
        
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def train():
    print(f"🚀 Loading model from: {MODEL_PATH}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Dataset
    train_dataset = load_and_format_dataset(DATA_FILE)
    print(f"✅ Loaded {len(train_dataset)} preference pairs.")

    # 3. Define Training Arguments (Optimized for RTX 4060 8GB)
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,                       # The "temperature" of DPO
        
        # Memory Optimization Settings
        per_device_train_batch_size=2,  # Keep this low (1 or 2)
        gradient_accumulation_steps=4,  # Accumulate gradients
        fp16=True,                      # Mixed Precision
        
        # Training Speed Settings
        learning_rate=5e-6,             # Very low LR for DPO
        max_steps=1000,                 
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
        
        # --- MOVED ARGUMENTS (Fixes the API Error) ---
        max_length=512,                 # Max sequence length (Prompt + Response)
        max_prompt_length=128,          # Max prompt length
    )

    # 4. Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=MODEL_PATH,               
        ref_model=None,                 # TRL will auto-load a reference copy
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,     # <--- RENAMED from 'tokenizer' to 'processing_class'
    )

    # 5. Start Training
    print("🔥 Starting DPO Training...")
    dpo_trainer.train()
    
    # 6. Save the final aligned model
    print(f"💾 Saving model to {OUTPUT_DIR}")
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer too for easy loading later

if __name__ == "__main__":
    train()