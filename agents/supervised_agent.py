import torch
import numpy as np
from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, execute_move
from train_supervised_agent import OthelloNet, BOARD_SIZE

MODEL_PATH = 'supervised_model.pth'

@register_agent('supervised_agent')
class SupervisedAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = 'SupervisedAgent'
        self.autoplay = True
        self.model = OthelloNet(BOARD_SIZE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.eval()
        self.temperature = 0.5

    def encode_board(self, board, player):
        arr = np.zeros((2, board.shape[0], board.shape[1]), dtype=np.float32)
        arr[0] = (board == player)
        arr[1] = (board == 3 - player)
        return torch.tensor(arr).unsqueeze(0)

    def evaluate_move(self, board, move, player):
        """Evaluate a move by simulating it and counting captured pieces"""
        new_board = board.copy()
        execute_move(new_board, move, player)
        return np.sum(new_board == player) - np.sum(board == player)

    def step(self, chess_board, player, opponent):
        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None
        
        inp = self.encode_board(chess_board, player)
        with torch.no_grad():
            logits = self.model(inp).cpu().numpy().flatten()

        logits = logits / self.temperature
        
        mask = np.full_like(logits, -np.inf)
        board_size = chess_board.shape[0]
        for r, c in valid_moves:
            mask[r * board_size + c] = logits[r * board_size + c]

        probs = np.exp(mask - np.max(mask))
        probs = probs / np.sum(probs)

        move_scores = []
        for r, c in valid_moves:
            move_idx = r * board_size + c
            model_score = probs[move_idx]
            capture_score = self.evaluate_move(chess_board, (r, c), player)
            combined_score = 0.7 * model_score + 0.3 * (capture_score / 10)
            move_scores.append((combined_score, (r, c)))

        best_move = max(move_scores, key=lambda x: x[0])[1]
        return best_move 