from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Loading model/tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fixing padding token (Very important for GPT-2)
tokenizer.pad_token = tokenizer.eos_token

# Loading Dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Creating Data Collator for Language Modeling (No Masked Language Modeling for GPT-2)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-medium-story",
    num_train_epochs=1,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    gradient_accumulation_steps=8,  # To simulate a larger batch size
    fp16=True,                      # Use mixed precision if available
    logging_steps=50,
    save_steps=500,
)

# Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,    
)

# Train and Save
print("Starting training on GPU...")
trainer.train()

print("Saving model...")
model.save_pretrained("./gpt2-story-finetuned")
tokenizer.save_pretrained("./gpt2-story-finetuned") # Always save the tokenizer too!
print("Done!")