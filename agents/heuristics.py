import numpy as np
from helpers import get_valid_moves, execute_move

class Heuristics:
    def __init__(self, weights=None):
        self.weights = weights or {
            "score": 2,          
            "mobility": 20,      
            "corner": 100,        
            "stability": 40,     
            "parity": 10,
            "edge_score": 10       
        }
        self.corner_positions = []
        self.edge_positions = []

    def evaluate_board(self, board, player, opponent, get_valid_moves):
        player_score, opponent_score = self.calculate_scores(board, player, opponent)
        player_moves, opponent_moves = self.calculate_mobility(board, player, opponent, get_valid_moves)
        corner_score = self.corner_heuristic(board, player) - self.corner_heuristic(board, opponent)
        stability_score = self.stability_heuristic(board, player) - self.stability_heuristic(board, opponent)
        edge_score_player = self.edge_heuristic(board, player)
        edge_score_opponent = self.edge_heuristic(board, opponent)
        edge_score = edge_score_player - edge_score_opponent
        parity_score = self.parity_heuristic(board)

        return (
            self.weights["score"] * (player_score - opponent_score) +
            self.weights["mobility"] * (player_moves - opponent_moves) +
            self.weights["corner"] * corner_score +
            self.weights["stability"] * stability_score +
            self.weights.get("edge", 0) * edge_score +
            self.weights["parity"] * parity_score
        )


    def calculate_scores(self, board, player, opponent):
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)
        return player_score, opponent_score

    def calculate_mobility(self, board, player, opponent, get_valid_moves):
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        return player_moves, opponent_moves

    def corner_heuristic(self, board, player):
        return sum(1 for (r, c) in self.corner_positions if board[r, c] == player)
    
    def update_corner_positions(self, board):
        rows, cols = board.shape
        self.corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    
    def edge_heuristic(self, board, player):
        return sum(1 for r, c in self.edge_positions if board[r, c] == player)

    def update_edge_positions(self, board):
        rows, cols = board.shape
        self.edge_positions = [(0, i) for i in range(1, cols - 1)] + \
                              [(rows - 1, i) for i in range(1, cols - 1)] + \
                              [(i, 0) for i in range(1, rows - 1)] + \
                              [(i, cols - 1) for i in range(1, rows - 1)]

    def stability_heuristic(self, board, player):
        stable_count = 0
        for r, c in np.argwhere(board == player):
            if self.is_stable(board, r, c, player):
                stable_count += 1
        return stable_count

    def is_stable(self, board, r, c, player):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Hàng và cột
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Đường chéo

        stable_directions = 0
        for dr, dc in directions:
            if self.is_direction_stable(board, r, c, dr, dc, player):
                stable_directions += 1
            if stable_directions >= 3:  # Nếu đủ 3 hướng ổn định, trả về True
                return True
        return False
        
    def is_direction_stable(self, board, r, c, dr, dc, player):
        rows, cols = len(board), len(board[0])
        current_r, current_c = r + dr, c + dc

        while 0 <= current_r < rows and 0 <= current_c < cols:
            if board[current_r, current_c] == 0:  # Gặp ô trống
                return False
            if board[current_r, current_c] != player:  # Gặp quân đối thủ
                return False
            current_r += dr
            current_c += dc
        return True  # Tất cả các ô đều là quân của player

    def parity_heuristic(self, board):
        empty_cells = np.sum(board == 0)
        return empty_cells if empty_cells % 2 == 0 else -empty_cells

    def score_move(self, board, move, player, execute_move_fn):
        # Kiểm tra nếu nước đi không hợp lệ
        valid_moves = get_valid_moves(board, player)
        if move not in valid_moves:
            return -float("inf")

        # Ưu tiên nước đi ở góc
        if move in self.corner_positions:
            return float("inf")  # Nước đi ở góc có giá trị cao nhất

        # Tránh nước đi gần góc
        near_corner_positions = [(0, 1), (1, 0), (1, 1), (0, board.shape[1] - 2), (1, board.shape[1] - 1),
                                (board.shape[0] - 2, 0), (board.shape[0] - 1, 1), (board.shape[0] - 2, 1),
                                (board.shape[0] - 2, board.shape[1] - 2), (board.shape[0] - 1, board.shape[1] - 2)]
        if move in near_corner_positions:
            return -float("inf")  # Nước đi gần góc có giá trị thấp nhất

        # Sao chép bàn cờ và thực hiện nước đi
        board_copy = board.copy()
        execute_move_fn(board_copy, move, player)

        # Đánh giá bàn cờ sau khi thực hiện nước đi
        opponent = 3 - player
        return self.evaluate_board(board_copy, player, opponent, get_valid_moves)