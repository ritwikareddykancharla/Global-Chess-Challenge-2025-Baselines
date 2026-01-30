import os
os.environ["WANDB_MODE"] = "disabled"  # hard-disable wandb on Trainium

# ================================
# Neuron / XLA environment settings
# ================================
os.makedirs("./neuron_cache", exist_ok=True)
os.environ["NEURON_COMPILE_CACHE_URL"] = "./neuron_cache"
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --auto-cast=none"
os.environ["XLA_USE_BF16"] = "1"

import argparse
import torch
import pyarrow.parquet as pq
from datasets import Dataset
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
parser = argparse.ArgumentParser(description="Fine-tune LLM on chess dataset (Trainium)")
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
assert OUTPUT_DIR.strip(), "Output directory must be non-empty"
if os.path.exists(OUTPUT_DIR):
    assert not os.listdir(OUTPUT_DIR), f"Output directory '{OUTPUT_DIR}' must be empty"

# ================================
# Config
# ================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_PATH = "./data/ChessExplained_2500k_qwen3.parquet"
TOKENIZER_PATH = os.path.abspath("./data_preparation/chess_tokenizer_qwen3/")

assert os.path.exists(DATASET_PATH), "Parquet dataset not found"
assert os.path.exists(TOKENIZER_PATH), "Tokenizer path not found"

NUM_LINES_TO_LOAD = 10_000
MAX_LENGTH = 512

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 500

EVAL_STEPS = 3000
SAVE_STEPS = 3000
SAVE_TOTAL_LIMIT = 3

TENSOR_PARALLEL_SIZE = 2

print("=" * 80)
print("Training on AWS Trainium trn1.2xlarge")
print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
print(f"Neuron Cache: {os.environ['NEURON_COMPILE_CACHE_URL']}")
print("=" * 80)

# ================================
# Dataset (flatten INSIDE train.py using raw PyArrow)
# ================================
print("Loading parquet dataset with PyArrow (raw)…")

table = pq.read_table(DATASET_PATH)
rows = table.to_pylist()

print("Flattening messages → text")
texts = []
for row in rows[:NUM_LINES_TO_LOAD]:
    messages = row.get("messages", [])
    text = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        text += f"{role}: {content}\n"
    texts.append({"text": text})

dataset = Dataset.from_list(texts)
print(f"Loaded and flattened {len(dataset)} examples")

# ================================
# Tokenizer (FORCE SLOW TOKENIZER – fixes Qwen tokenizer.json crash)
# ================================
print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    use_fast=False,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================================
# Training arguments
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
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to="none",
    dataloader_drop_last=True,
)

training_config = NeuronSFTConfig(
    packing=True,
    **training_args.to_dict(),
)

# ================================
# Prepare base model (resize embeddings)
# ================================
PREPARED_MODEL_DIR = MODEL_NAME.replace("/", "_") + "_prepared"

if not os.path.exists(PREPARED_MODEL_DIR):
    print("Preparing base model with resized embeddings…")
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
# Load Neuron model
# ================================
print("Loading Neuron model (first compile will be slow)…")
model = NeuronModelForCausalLM.from_pretrained(
    PREPARED_MODEL_DIR,
    training_args.trn_config,
    dtype=torch.bfloat16,
)

print(f"Model parameters: {model.num_parameters():,}")

# ================================
# Formatting function
# ================================
def format_chess_dataset(example):
    return example["text"]

# ================================
# Trainer
# ================================
trainer = NeuronSFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    eval_dataset=None,
    processing_class=tokenizer,
    formatting_func=format_chess_dataset,
)

print("NeuronSFTTrainer initialized")
print("⚠️ First run will compile Neuron graphs (slow but expected)")

# ================================
# Train
# ================================
try:
    trainer.train()
    print("✅ Training complete")

    final_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

except KeyboardInterrupt:
    print("Training interrupted — saving checkpoint")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted_checkpoint"))

finally:
    print("Training session ended")
