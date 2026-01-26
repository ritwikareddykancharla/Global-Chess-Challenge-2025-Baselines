import os
import torch
import argparse
from datasets import Dataset
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # Top choice for <8B
TOKENIZER_PATH = "./data_preparation/chess_tokenizer_qwen3/"
DATASET_PATH = "./data/ChessExplained_2500k_qwen3.parquet"
OUTPUT_DIR = "./trained_models/qwen7b_chess_sft"

def main():
    # 1. Load Tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model (Full bfloat16 for H100)
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load & Tokenize Data
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = Dataset.from_parquet(DATASET_PATH).select(range(500_000)) # Start with 500k

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8, # H100 can handle this easily
        gradient_accumulation_steps=2,
        learning_rate=5e-5, # Lower LR for 7B models
        weight_decay=0.01,
        bf16=True,
        logging_steps=100,
        save_strategy="epoch",
        report_to="wandb"
    )

    # 5. Start Training
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    
    final_path = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Training complete! Model saved to {final_path}")

if __name__ == "__main__":
    main()
