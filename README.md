# ♟️ Chess LLM Baselines

Baselines for the [Global Chess Challenge 2025](https://www.aicrowd.com/challenges/global-chess-challenge-2025)

Supervised fine-tuning (SFT) for chess using special tokens to encode board positions. This approach trains a language model to predict moves given structured chess position representations.

## 🎯 Rationale for Special Tokens

Traditional chess representations like FEN strings have tokenization issues:
- ❌ **FEN strings**: Characters get merged unpredictably by BPE tokenizers (e.g., "rnbq" might become one token), making it hard for the model to learn piece-by-piece understanding
- ❌ **ASCII board representations**: No tokenization issues, but inefficient (~350-400+ tokens per position)

**✨ Special token encoding solves both problems:**
- ✅ Each piece and square is a single, unambiguous token (e.g., `<a1><White_Rook>`, `<e5><Black_Pawn>`)
- ✅ Compact representation (~130-170 tokens per position including legal moves)
- 🚀 **Results**: Significantly fewer illegal moves during training and faster convergence due to reduced sequence length

### 📝 Encoding Format Example

```
<chess_position>
<a1><White_Rook><b1><White_Knight>...<h8><Black_Rook>
|White|KQkq|-|0|1|
<e2><e4> <g1><f3> <b1><c3> ...
</chess_position>
```

**Structure:** `[64 square-piece pairs] | [turn] | [castling] | [en passant] | [halfmove] | [fullmove] | [legal moves]`

💡 See `data_preparation/encode_with_special_tokens.ipynb` for detailed implementation.

---

## 🚀 Quick Start

1. **Download the dataset**: `cd data && bash download_data.sh`
2. **Train the model**: `python train.py --output-dir ./trained_models/your_model_name`

## 🔧 Dataset Preparation (Optional)

Modify encoding schemes or add custom prompting:
- `prepare_tokenizer.ipynb` - Create a tokenizer with added special tokens for chess
- `encode_with_special_tokens.ipynb` - Encode chess positions using special tokens

---

## 🏋️ Training

The training script (`train.py`) fine-tunes Qwen3-0.6B on chess positions with a custom tokenizer that includes special tokens for pieces and squares.

**Training Pipeline:**
1. Loads a custom tokenizer with added chess special tokens (`chess_tokenizer_qwen3/`)
2. Resizes model embeddings to accommodate new tokens
3. Tokenizes dataset with causal language modeling objective
4. Trains with gradient accumulation and bfloat16 precision
5. Evaluates during training via `ChessLLMEvaluationCallback` (puzzles, vs random, vs Stockfish)

**Evaluation During Training:**

The `ChessLLMEvaluationCallback` automatically evaluates your model at regular intervals (e.g., every 3000 steps). The callback:
- Saves a temporary checkpoint of the current model
- Moves the model to CPU and frees GPU memory
- Starts a vLLM server with the checkpoint for fast inference
- Runs the full evaluation suite (puzzle solving, games vs random player, games vs Stockfish)
- Logs metrics to WandB and shuts down vLLM
- Restores the model to GPU and resumes training

You can modify this callback to add any other metrics you want to track.

**Key Hyperparameters:**

- **`MAX_LENGTH`** (512): Maximum sequence length. Chess positions with special tokens are compact (~130-170 tokens), so 512 is sufficient.
- **`BATCH_SIZE`** (4) + **`GRAD_ACCUM_STEPS`** (2): Effective batch size of 8. Increase `BATCH_SIZE` if you have more GPU memory (24GB+ allows batch size 8-16). Adjust `GRAD_ACCUM_STEPS` to maintain effective batch size.
- **`LEARNING_RATE`** (1e-4): Standard for SFT on small models. If training is unstable, reduce to 5e-5. If convergence is slow, try 2e-4.
- **`WARMUP_STEPS`** (500): Gradual learning rate warmup to stabilize early training. Typically 5-10% of total steps.
- **`WEIGHT_DECAY`** (0.001): L2 regularization to prevent overfitting on the training set.
- **`NUM_LINES_TO_LOAD`** (1M): Number of training examples from the 2.5M dataset. Start with fewer for faster iteration (100k-500k).
- **`EVAL_STEPS`** (3000): How often to run full evaluation (puzzles + games). Evaluation takes ~10-15 minutes, so balance between frequent feedback and training speed.

---

## 🎮 Evaluation

Run `python run_evaluation.py` after starting a vLLM server with your trained model. The evaluation suite measures chess playing strength across three dimensions.

**Setup:**
1. Start vLLM server: `bash run_vllm.sh` (edit the script to point to your model checkpoint) in a separate terminal
2. Run evaluation: `python run_evaluation.py -v`
3. Results saved to `eval_results/` directory with JSON metrics and PGN game files

**Evaluation Metrics** (configured in `evaluation_helpers/eval_config.py`):

1. 🧩 **Puzzle Solving** (`n_puzzles=200`, `puzzle_max_elo=600`)
   - Tests tactical pattern recognition on chess puzzles
   - Measures: solve rate, average moves to solution
   - Easy puzzles (Elo ≤600) test basic tactics and checkmates
   
2. 🎲 **vs Random Player** (`n_random_games=5`)
   - Sanity check that model can beat random move selection
   - Should achieve 100% win rate quickly
   - Measures: win rate, average game length, illegal move rate, ACPL
   
3. 🤖 **vs Stockfish** (`n_stockfish_games=50`, `stockfish_depth=1`, `stockfish_skill_level=0`)
   - Tests against weak but coherent opponent
   - Stockfish at depth 1, skill 0 plays at ~1000-1200 Elo
   - Measures: win/draw/loss rates, illegal moves, ACPL

**LLM Inference Settings:**
- **`temperature=0.0`**: Deterministic, always picks highest probability move
- **`max_retries=3`**: If model generates illegal move, retry with same or slightly higher temperature