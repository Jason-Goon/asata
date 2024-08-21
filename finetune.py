import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Limit PyTorch's use of threads to avoid memory spikes
torch.set_num_threads(1)

# Load a small subset of your text data for testing
dataset = load_dataset('text', data_files={'train': 'context.txt'}, split='train[:10%]')

# Verify dataset size
print(f"Dataset size: {len(dataset)} examples")

# Load the model and tokenizer using the correct local path
tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/llama-3b-finnish")
model = AutoModelForCausalLM.from_pretrained("Finnish-NLP/llama-3b-finnish")

# Set the pad_token to be the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)  # Reduce max_length further
    tokens['labels'] = tokens['input_ids'].copy()  # Set labels to input_ids for language modeling
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator that handles dynamic padding and label shifting for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with further reduced resource usage
training_args = TrainingArguments(
    output_dir="./llama_results",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,  # Keep batch size very low
    gradient_accumulation_steps=1,  # Reduce accumulation steps to minimize memory usage
    num_train_epochs=1,  # Use just one epoch for testing
    save_steps=50,  # Save less frequently
    save_total_limit=1,
    logging_dir="./llama_logs",
    fp16=False,  # Disable mixed precision training
    no_cuda=True,  # Force CPU-only
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_llama_finnish_model")
