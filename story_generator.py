from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. Load model/tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fix padding token (Crucial for GPT-2)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load Dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

def tokenize(examples):
    # CHANGED: Removed padding="max_length" here to let DataCollator handle it dynamically (Faster)
    return tokenizer(examples["text"], truncation=True, max_length=512)

# CHANGED: Added remove_columns=["text"] to keep the dataset clean for the trainer
tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 3. Create Data Collator (This fixes the "No Loss" error)
# It automatically copies input_ids to labels for you
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-medium-story",
    num_train_epochs=1,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    gradient_accumulation_steps=8,  # To simulate a larger batch size
    fp16=True,                      # ENABLED: Huge speed boost for your RTX 4060
    logging_steps=50,
    save_steps=500,
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,    # <--- Passed the collator here
)

# 6. Train and Save
print("Starting training on GPU...")
trainer.train()

print("Saving model...")
model.save_pretrained("./gpt2-story-finetuned")
tokenizer.save_pretrained("./gpt2-story-finetuned") # Always save the tokenizer too!
print("Done!")