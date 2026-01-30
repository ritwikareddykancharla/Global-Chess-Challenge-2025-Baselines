import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Neuron/XLA optimizations - set before importing torch
os.makedirs("./neuron_cache", exist_ok=True)
os.environ["NEURON_COMPILE_CACHE_URL"] = "./neuron_cache"
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --auto-cast=none"
os.environ["XLA_USE_BF16"] = "1"

import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

# Import from optimum.neuron (compatible with 0.0.24)
from optimum.neuron import NeuronTrainer, NeuronTrainingArguments

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune LLM on chess dataset")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./trained_models/chess_qwen_finetuned",
    help="Directory to save the fine-tuned model and checkpoints"
)
args = parser.parse_args()

# Validate output directory argument
assert args.output_dir and args.output_dir.strip(), "Output directory must be provided and non-empty"
if os.path.exists(args.output_dir):
    if len(os.listdir(args.output_dir)) > 0:
        print(f"Warning: Output directory '{args.output_dir}' already exists and is not empty")

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "aicrowd/ChessExplained"
OUTPUT_DIR = args.output_dir
NUM_LINES_TO_LOAD = 10000

MAX_LENGTH = 512  

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
WARMUP_STEPS = 500

# Logging and checkpointing
LOGGING_STEPS = 100
SAVE_STEPS = 3000
SAVE_TOTAL_LIMIT = 3

# Trainium-specific settings for trn1.2xlarge (2 NeuronCores)
TENSOR_PARALLEL_SIZE = 2

# Run name for logging
run_name = os.path.basename(os.path.normpath(OUTPUT_DIR))

print("="*80)
print("Training on AWS Trainium trn1.2xlarge")
print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Neuron Cache: {os.environ['NEURON_COMPILE_CACHE_URL']}")
print("="*80)

# Load dataset
print(f"Loading dataset from HuggingFace: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.select(range(min(NUM_LINES_TO_LOAD, len(dataset))))
print(f"Loaded {len(dataset)} examples")

# Load tokenizer
print(f"Loading tokenizer from model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Model loaded with {model.num_parameters():,} parameters")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments (NeuronTrainingArguments for Trainium)
training_args = NeuronTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    bf16=True,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_drop_last=True,
    num_train_epochs=1,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
)

print("\n" + "="*80)
print("Training Configuration:")
print("="*80)
print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Weight decay: {training_args.weight_decay}")
print(f"  Warmup steps: {training_args.warmup_steps}")
print(f"  Total epochs: {training_args.num_train_epochs}")
print("="*80 + "\n")

# Initialize NeuronTrainer
trainer = NeuronTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("NeuronTrainer initialized. Ready to start training.")
print("\n⚠️  IMPORTANT: First run will compile graphs. Be patient!")

try:
    # Start training
    print("Starting training...")
    print("Monitor Neuron utilization with: neuron-top\n")

    trainer.train()

    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80 + "\n")

    # Save the final model and tokenizer
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"Saving final model to {final_model_path}")

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"\n✅ Model and tokenizer saved to {final_model_path}")
    print(f"\nTo load the model:")
    print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{final_model_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{final_model_path}")')
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving current checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted_checkpoint")
    print("Checkpoint saved.")
    
finally:
    print("\nTraining session ended.")