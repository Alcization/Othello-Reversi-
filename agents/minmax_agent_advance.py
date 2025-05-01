from agents.agent import Agent
from store import register_agent
import numpy as np # type: ignore
from copy import deepcopy
import time
from agents.danh_heuristics import *


@register_agent("minmax_agent_advance")
class MinMaxAgentAdvance(Agent):
    def __init__(self):
        super(MinMaxAgentAdvance, self).__init__()
        self.name = "MinMaxAgentAdvance"
        self.time_limit = 2
        self.depth = 3

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        best_move = None
        _, move = self.alphabeta(chess_board, player, opponent, self.depth, True, start_time=start_time)
        if move is not None:
            best_move = move
        return best_move

    def alphabeta(self, chess_board, player, opponent, depth, isMax, alpha=float('-inf'), beta=float('inf'), start_time=None):
        if time.time() - start_time >= self.time_limit:
            return 0, None

        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            return self.evaluate_board(chess_board, player, opponent), None

        legal_moves = get_valid_moves(chess_board, player if isMax else opponent)
        if not legal_moves:
            return self.alphabeta(chess_board, player, opponent, depth - 1, not isMax, alpha, beta, start_time)
        
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

    def evaluate_board(self, chess_board: np.ndarray, player, opponent) -> float:
        score = evaluate_score_comparison(evaluate_player_current_score(chess_board, player),
                                          evaluate_player_current_score(chess_board, player))

        mobility = evaluate_score_comparison(evaluate_mobility(chess_board, player),
                                             evaluate_mobility(chess_board, opponent))

        corners = evaluate_score_comparison(
            count_corner_stable_disks(chess_board, player, 0),
            count_corner_stable_disks(chess_board, opponent, 0))

        stable_disks = evaluate_score_comparison(
            evaluate_stable_disks(chess_board, player),
            evaluate_stable_disks(chess_board, opponent))

        final_score = score * 0.1 + mobility * 0.1 + corners * 0.6 + stable_disks * 0.2
        return final_score
