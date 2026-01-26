# ♟️ Global Chess Challenge 2025: Advanced Strategies

This document outlines the advanced strategies and training techniques for moving from the SFT baseline to a competitive leaderboard position.

---

## 🚀 Scaling and Hardware (H100 / Kaggle)

For high-performance training on H100 or Kaggle:
- **Model**: Use **Qwen2.5-7B-Instruct**. It is the strongest model under the 8B parameter limit.
- **Precision**: Fine-tune in `bfloat16` to maintain accuracy while saving memory.
- **Backend**: Use `vLLM` for fast inference during evaluation.

### Installation
```bash
pip install git+https://github.com/AIcrowd/chess-env.git
sudo apt update && sudo apt install -y stockfish
```

---

## 📊 Supervised Fine-Tuning (SFT) Strategies

### 1. 🏆 Elo Conditioning
Prepended an Elo tag to every training example to help the model learn the difference between "bad" moves and "Grandmaster" moves.
- **Tag**: `[Elo: Grandmaster]`
- **Inference**: Always prompt with this tag to force the model into its best player persona.

### 2. 🧠 Chain-of-Thought (CoT)
Train on sequences that include reasoning before the move:
```text
<chess_position>...
<think>The opponent's King is exposed, I should calculate a mating sequence starting with Nxf7.</think>
<uci_move>g5f7</uci_move>
```

### 3. 🧪 "Gold" Data Filtering
Only train on moves where the engine evaluation change is minimal (< 30 centipawns). Blunders in the training set will confuse the model.

---

## 🧠 Group Relative Policy Optimization (GRPO)

GRPO allows the model to learn from "verifiable rewards" (legality, Stockfish evals) without a critic model.

### Recommended Rewards:
1.  **Legality Reward**: +1.0 for legal moves, -1.0 for illegal/malformed.
2.  **Format Reward**: +0.5 for strictly following `<uci_move>` and `<rationale>` tags.
3.  **Strength Reward**: Scalar reward based on Stockfish evaluation change (CP or WinProb).

Refer to `grpo_train_template.py` for a starting implementation.
