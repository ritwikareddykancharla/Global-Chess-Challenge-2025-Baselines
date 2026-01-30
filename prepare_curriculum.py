import argparse
import chess
import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm

# --- Configuration ---
# You can adjust these
OUTPUT_DIR = "./data_curriculum"
MAX_SAMPLES = 100000

def fen_to_xml(fen_str):
    """
    Converts a FEN string to the custom <chess_position> XML format.
    Adaptation of the Jinja logic.
    """
    try:
        board = chess.Board(fen_str)
        xml = "<chess_position>"
        
        # 1. Squares
        # Iterating a1 (0) to h8 (63)
        piece_map_rev = {
            'P': '<White_Pawn>', 'N': '<White_Knight>', 'B': '<White_Bishop>',
            'R': '<White_Rook>', 'Q': '<White_Queen>', 'K': '<White_King>',
            'p': '<Black_Pawn>', 'n': '<Black_Knight>', 'b': '<Black_Bishop>',
            'r': '<Black_Rook>', 'q': '<Black_Queen>', 'k': '<Black_King>'
        }
        
        for square in chess.SQUARES:
            sq_name = chess.square_name(square)
            piece = board.piece_at(square)
            if piece:
                piece_code = piece.symbol()
                token = piece_map_rev.get(piece_code, '<blank>')
            else:
                token = '<blank>'
            xml += f"<{sq_name}>{token}"
            
        xml += "|"
        
        # 2. Metadata
        xml += "White" if board.turn == chess.WHITE else "Black"
        xml += f"|{board.castling_rights}|-|0|1|" # Simplified metadata for basic training
        
        # 3. Legal Moves (Optional, heavy)
        # xml += " ".join([m.uci() for m in board.legal_moves])
        
        xml += "</chess_position>"
        return xml
    except:
        return None

def process_puzzles(num_samples=MAX_SAMPLES):
    """
    Downloads Lichess puzzles and converts them to SFT format.
    Format:
    User: <XML_BOARD> Solve this puzzle.
    Assistant: <think>...</think> <uci_move>BEST_MOVE</uci_move>
    """
    print(f"⏳ Processing {num_samples} puzzles...")
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    
    data = []
    
    for i, row in tqdm(enumerate(ds), total=num_samples):
        if i >= num_samples: break
        
        # Puzzles provide 'fen' usually, or 'ctx' (pgn). 
        # Check schemas carefully. EleutherAI version has 'fen' usually.
        # If 'ctx' (moves), we must replay.
        
        # Fallback if fen not present (common in raw dataset)
        fen = row.get('fen', None)
        target_move = row.get('target', '') # "e2e4"
        
        if not fen and 'ctx' in row:
            # Replay moves (ctx is usually space separated moves)
            board = chess.Board()
            try:
                for move in row['ctx'].split():
                    board.push_san(move)
                fen = board.fen()
            except:
                continue
                
        if fen and target_move:
            xml_board = fen_to_xml(fen)
            if xml_board:
                data.append({
                    "prompt": f"{xml_board}\nUser: Find the winning move.",
                    "completion": f"<think>The winning tactic involves...</think>\n<uci_move>{target_move}</uci_move>"
                })
    
    df = pd.DataFrame(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(f"{OUTPUT_DIR}/train_puzzles.parquet")
    print(f"✅ Saved {len(df)} puzzles to {OUTPUT_DIR}/train_puzzles.parquet")

def process_evals(num_samples=MAX_SAMPLES):
    """
    Downloads Fishnet Evals and converts.
    """
    print(f"⏳ Processing {num_samples} evaluations...")
    # Use streaming to handle the massive dataset
    ds = load_dataset("Lichess/fishnet-evals", split="train", streaming=True)
    
    data = []
    for i, row in tqdm(enumerate(ds), total=num_samples):
        if i >= num_samples: break
        
        fen = row['fen']
        score = row['cp'] # Centipawn
        best_line = row['line'].split()[0] if row['line'] else '' # First move of PV
        
        if fen and best_line:
            xml_board = fen_to_xml(fen)
            if xml_board:
                # We can add score to prompt to teach "Evaluation"
                data.append({
                    "prompt": f"{xml_board}\nUser: Evaluate the position.",
                    "completion": f"<think>Evaluation: {score} cp.</think>\n<uci_move>{best_line}</uci_move>"
                })
                
    df = pd.DataFrame(data)
    df.to_parquet(f"{OUTPUT_DIR}/train_evals.parquet")
    print(f"✅ Saved {len(df)} evaluations to {OUTPUT_DIR}/train_evals.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles", action="store_true", help="Process Puzzles")
    parser.add_argument("--evals", action="store_true", help="Process Fishnet Evals")
    args = parser.parse_args()
    
    if args.puzzles: process_puzzles()
    if args.evals: process_evals()
