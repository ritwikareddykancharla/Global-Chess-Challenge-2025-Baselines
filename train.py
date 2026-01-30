import os

# ================================
# Neuron / XLA environment settings
# ================================
os.makedirs("./neuron_cache", exist_ok=True)
os.environ["NEURON_COMPILE_CACHE_URL"] = "./neuron_cache"
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --auto-cast=none"
os.environ["XLA_USE_BF16"] = "1"

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.trainers import (
    NeuronSFTTrainer,
    NeuronSFTConfig,
    NeuronTrainingArguments,
)

# ================================
# CLI args
# ================================
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
assert not os.path.exists(OUTPUT_DIR), "Output directory must not already exist"

# ================================
# Config
# ================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "aicrowd/ChessExplained"
NUM_LINES_TO_LOAD = 10_000

MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 500
TENSOR_PARALLEL_SIZE = 2

# ================================
# Dataset
# ================================
print("Loading dataset…")
dataset = load_dataset(DATASET_NAME, split="train")
if NUM_LINES_TO_LOAD:
    dataset = dataset.select(range(min(NUM_LINES_TO_LOAD, len(dataset)))))

# ================================
# Tokenizer
# ================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================================
# Prepare base model (resize embeddings)
# ================================
PREPARED_MODEL_DIR = MODEL_NAME.replace("/", "_") + "_prepared"

if not os.path.exists(PREPARED_MODEL_DIR):
    print("Preparing base model…")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.save_pretrained(PREPARED_MODEL_DIR)
    tokenizer.save_pretrained(PREPARED_MODEL_DIR)
    del base_model

# ================================
# Tokenization
# ================================
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
)

# ================================
# Training args
# ================================
training_args = NeuronTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    logging_steps=100,
    save_steps=3000,
    save_total_limit=3,
    report_to="none",
    dataloader_drop_last=True,
)

training_config = NeuronSFTConfig(
    packing=True,
    **training_args.to_dict(),
)

# ================================
# Load Neuron model
# ================================
print("Loading Neuron model (first compile is slow)…")
model = NeuronModelForCausalLM.from_pretrained(
    PREPARED_MODEL_DIR,
    training_args.trn_config,
    dtype=torch.bfloat16,
)

# ================================
# Trainer
# ================================
trainer = NeuronSFTTrainer(
    model=model,
    args=training_config,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
)

# ================================
# Train
# ================================
print("Starting training 🚀")
trainer.train()

# ================================
# Save
# ================================
final_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print("✅ Training complete and model saved")
