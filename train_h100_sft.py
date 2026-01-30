import os
# os.environ["WANDB_MODE"] = "disabled"

import torch
import pandas as pd
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
from chess_evaluation_callback import ChessLLMEvaluationCallback

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune LLM on chess dataset (H100 Optimized)")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./trained_models/qwen7b_chess_finetuned",
    help="Directory to save the fine-tuned model and checkpoints"
)
args = parser.parse_args()

# Validate output directory argument
assert args.output_dir and args.output_dir.strip(), "Output directory must be provided and non-empty"
if os.path.exists(args.output_dir):
    assert len(os.listdir(args.output_dir)) == 0, f"Output directory '{args.output_dir}' already exists and is not empty"

# Configuration - H100 UPGRADES HERE
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # <--- 7B Model
DATASET_PATH = "./data/ChessExplained_2500k_qwen3.parquet"
assert os.path.exists(DATASET_PATH), f"Dataset file '{DATASET_PATH}' does not exist, please run the download script in the data directory"
TOKENIZER_PATH = "./data_preparation/chess_tokenizer_qwen3/"
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)
OUTPUT_DIR = args.output_dir
NUM_LINES_TO_LOAD = 1_000_000
# Maximum sequence length, for the dataset with special tokens, the sequences are short so 512 is enough
MAX_LENGTH = 512  
EVAL_STEPS = 3000 # Evaluate every N steps

BATCH_SIZE = 8 # <--- H100 Batch Size
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 5e-5 # <--- Slower LR for 7B
WEIGHT_DECAY = 0.001
WARMUP_STEPS = 500
LOGGING_STEPS = EVAL_STEPS
SAVE_STEPS = EVAL_STEPS
SAVE_TOTAL_LIMIT = 10
NUM_TRAIN_EPOCHS = 1

# Extract directory name for wandb run name
run_name = os.path.basename(os.path.normpath(OUTPUT_DIR))
wandb.init(project="ChessLLM", name=run_name)

print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# %%
dataset = Dataset.from_parquet(DATASET_PATH).select(range(NUM_LINES_TO_LOAD))
print(f"Loaded {len(dataset)} examples")

# %%
# Load tokenizer and model
print(f"Loading model: {MODEL_NAME}")
print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    # MODEL_NAME,
    TOKENIZER_PATH,
    trust_remote_code=True,
    padding_side="right"
)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # <--- H100 Upgrade
)
model.resize_token_embeddings(len(tokenizer))

print(f"Model loaded with {model.num_parameters():,} parameters")

# %%
# Tokenize the dataset
def tokenize_function(examples):
    """Tokenize text examples"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # We'll pad dynamically during training
        return_tensors=None
    )
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

print(f"Tokenization complete. Sample token count: {len(tokenized_dataset[0]['input_ids'])}")


# %%
# Set up training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=False,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,  # Add this line
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    bf16=True,
    fp16=False,
    optim="adamw_torch_fused",
    remove_unused_columns=False,
    report_to="wandb",
    dataloader_drop_last=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
)

print("Training configuration:")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")

# %%
# Create data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Create evaluation callback
evaluation_callback = ChessLLMEvaluationCallback(
    model=model,
    tokenizer=tokenizer,
    checkpoint_dir=OUTPUT_DIR,
    eval_every_n_steps=EVAL_STEPS,
    output_dir="./eval_results_during_training",
    vllm_port=8000,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    warmup_steps=WARMUP_STEPS,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[evaluation_callback]
)

print("Trainer initialized with full evaluation callback. Ready to start training.")

# %%
try:
# Start training
    print("Starting training...")

    trainer.train()

    print("\n✅ Training complete!")

    # %%
    # Save the final model and tokenizer
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"Saving final model to {final_model_path}")

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"✅ Model and tokenizer saved to {final_model_path}")
    print(f"\nTo load the model:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{final_model_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{final_model_path}")')
except KeyboardInterrupt:
    print("Training interrupted by user")
finally:
    wandb.finish()
