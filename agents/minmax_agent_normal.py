from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import execute_move, check_endgame, get_valid_moves
from agents.heuristics import Heuristics  # Nhập lớp Heuristics

@register_agent("minmax_agent_normal")
class MinMaxAgentNormal(Agent):
    def __init__(self):
        super(MinMaxAgentNormal, self).__init__()
        self.name = "MinMaxAgentNormal"
        self.time_limit = 2
        self.depth = 4  # Tăng độ sâu để tận dụng heuristics
        self.heuristics = Heuristics()  # Khởi tạo Heuristics

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        best_move = None
        _, move = self.alphabeta(
            chess_board=chess_board, 
            player=player, 
            opponent=opponent, 
            depth=self.depth, 
            isMax=True, 
            start_time=start_time
        )
        if move is not None:
            best_move = move
        return best_move

    def alphabeta(self, chess_board, player, opponent, depth, isMax, alpha=float('-inf'), beta=float('inf'), start_time=None):
        if time.time() - start_time >= self.time_limit:
            return 0, None

        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            return self.heuristics.evaluate_board(chess_board, player, opponent, get_valid_moves), None

        legal_moves = get_valid_moves(chess_board, player if isMax else opponent)
        if not legal_moves:
            return self.alphabeta(chess_board, player, opponent, depth - 1, not isMax, alpha, beta, start_time)
        
        # Sắp xếp nước đi bằng heuristics
        legal_moves = self.sort_moves(chess_board, legal_moves, player if isMax else opponent)
        best_move = None

        if isMax:
            value = float('-inf')
            for move in legal_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, player)
                move_value, _ = self.alphabeta(new_board, player, opponent, depth - 1, False, alpha, beta, start_time)
                if move_value > value:
                    value = move_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        
        else:
            value = float('inf')
            for move in legal_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, opponent)
                move_value, _ = self.alphabeta(new_board, player, opponent, depth - 1, True, alpha, beta, start_time)
                if move_value < value:
                    value = move_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move

    def sort_moves(self, chess_board, moves, player):
        """Sắp xếp các nước đi dựa trên heuristic từ Heuristics."""
        move_scores = []
        for move in moves:
            # Ưu tiên góc
            if move in self.heuristics.corner_positions:
                score = float('inf')  # Góc có điểm cao nhất
            # Tránh nước đi gần góc
            elif move in [(1, 0), (0, 1), (1, 1), (6, 7), (7, 6), (6, 6)]:
                score = -float('inf')  # Nước đi gần góc có điểm thấp nhất
            else:
                score = self.heuristics.score_move(chess_board, move, player, execute_move)
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]