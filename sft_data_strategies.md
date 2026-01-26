# 📊 SFT Data Strategies: Winning Without RL

You can achieve "Grandmaster" reasoning using only Supervised Fine-Tuning (SFT) by carefully engineering your training data.

---

## 1. 🏆 Elo Conditioning
This is the most effective way to make a model "play better" without RL.
- **The Trick**: Prepended an Elo tag to every training example.
- **Example Prompt**:
  ```text
  [Elo: 2800] <chess_position>...
  ```
- **How to train**: Map the Elo of the players in your dataset to categories (e.g., `Beginner`, `Intermediate`, `Grandmaster`).
- **Inference**: Always use the `[Elo: Grandmaster]` tag in your prompt. The model will try to predict the move that a 2800+ player would make.

---

## 2. 🧠 Baking in Chain-of-Thought (CoT)
The challenge allows `<think>` or `<rationale>` tags. Use them in your training data!
- **Dataset Modification**: Don't just train on `[FEN] -> [Move]`. Train on:
  ```text
  <chess_position>...
  <think>I need to protect my King and control the d4 square.</think>
  <uci_move>e2e4</uci_move>
  ```
- **Where to get data?**: Use Stockfish to generate these rationales. You can prompt a large model (like GPT-4o) to "explain" Stockfish's best move in one sentence, then use that to train your smaller Qwen model.

---

## 3. 🧪 "Gold" Data Filtering (Stockfish Teacher)
Machine learning is "garbage in, garbage out." Most human games are full of blunders.
- **The Strategy**: Run your dataset through Stockfish (Depth 10 is enough). 
- **Filter Rule**: Only keep moves where the `Stockfish Eval Change < 30 centipawns`.
- **Result**: You are training the model to be a "mini-Stockfish" rather than a "average human player."

---

## 4. 🧩 Legal Move "Weighting"
To prevent illegal moves, format your prompt to include the legal moves at the end, and train the model to "pay attention" to them.
- **Template**:
  ```text
  Position: [Special Tokens]
  Turn: White
  Legal Moves: e2e4, d2d4, g1f3...
  Move: <uci_move>e2e4</uci_move>
  ```

---

## 🛠️ Implementation Script (Data Preprocessor)

```python
def preprocess_example(example, stockfish_evaluator):
    # 1. Get Elo of the winner
    elo = example['white_elo'] if example['winner'] == 'white' else example['black_elo']
    tag = "[Elo: GM]" if elo > 2500 else "[Elo: Club]"
    
    # 2. Extract best move from Stockfish (for "Gold" filtering)
    best_move = stockfish_evaluator.get_best_move(example['fen'])
    
    # 3. Create the SFT text
    text = f"{tag} <chess_position>{example['encoded_board']}</chess_position>"
    text += f"<think>Best move evaluated by engine.</think>"
    text += f"<uci_move>{best_move}</uci_move>"
    
    return {"text": text}
```

> [!TIP]
> **Data > Algorithm.** A 0.5B model trained on 1 million "Perfect Play" (Stockfish-approved) moves will easily beat a 7B model trained on messy human data.
