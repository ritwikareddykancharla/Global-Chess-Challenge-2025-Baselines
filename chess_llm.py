"""LLM utilities for chess evaluation"""

import chess
import re
from openai import OpenAI


class ChessLLM:
    """Chess LLM interface for move generation"""
    
    def __init__(self, config):
        self.config = config
        self.chess_template = config.chess_template
    
    def encode_board_position_jinja(self, fen):
        """Encode FEN to special token sequence using Jinja template.
        
        Args:
            fen: FEN string representing the board position
            
        Returns:
            Encoded board position string with special tokens
        """
        board = chess.Board(fen)
        
        # Get legal moves in UCI format
        legal_moves_uci = " ".join([move.uci() for move in board.legal_moves])
        
        # Render the template with FEN and legal moves
        encoded = self.chess_template.render(
            FEN=fen,
            legal_moves_uci=legal_moves_uci
        )
        
        return encoded.strip()
    
    def extract_uci_move(self, response):
        """Extract UCI move from model response"""
        try:
            match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
            if match:
                return match.group(1)
            
            # Try to find any UCI-like move pattern
            match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
            if match:
                return match.group(1)
            
            return "a1a1"

        except Exception:
            return "a1a1"
    
    def generate(self, messages):
        """Generate completion from vLLM server using OpenAI Python client
        
        Args:
            messages: List of message dicts, e.g. [{"role": "user", "content": "..."}]
        """
        client = OpenAI(
            base_url=f"http://localhost:{self.config.port}/v1",
            api_key="EMPTY"  # vLLM doesn't require a real API key
        )
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def get_move(self, board, temperature=None):
        """Get a move from the LLM for the current board position"""
        if temperature is None:
            temperature = self.config.temperature
            
        fen = board.fen()
        prompt = self.encode_board_position_jinja(fen)
        messages = [{"role": "user", "content": prompt}]
        response = self.generate(messages)
        
        # Extract thinking
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else None
        
        # Extract move
        move = self.extract_uci_move(response)
        
        return move, thinking
    
    def try_move(self, board):
        """Try to get a legal move from LLM with retries"""
        for attempt in range(self.config.max_retries):
            temperature = self.config.temperature if attempt == 0 else self.config.retry_temperature
            move_uci, thinking = self.get_move(board, temperature=temperature)
            
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move, thinking, False
            except (ValueError, AttributeError):
                pass
            except:
                import pdb; pdb.set_trace()
        
        return None, None, True  # illegal_move = True

