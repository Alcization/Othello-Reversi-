import numpy as np
from helpers import get_valid_moves, execute_move

class Heuristics:
    def __init__(self, weights=None):
        # Trọng số có thể được cấu hình
        self.weights = weights or {
            "score": 5,          # Trọng số cho điểm cơ bản
            "mobility": 15,      # Trọng số cho khả năng di chuyển
            "corner": 70,        # Trọng số cho kiểm soát góc
            "stability": 20,     # Trọng số cho ổn định cạnh
            "parity": 10         # Trọng số cho tính chẵn lẻ
        }
        self.corner_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
        self.edge_positions = [
            (0, i) for i in range(1, 7)
        ] + [(7, i) for i in range(1, 7)] + [(i, 0) for i in range(1, 7)] + [(i, 7) for i in range(1, 7)]

    def evaluate_board(self, board, player, opponent, get_valid_moves):
        # Tính điểm cơ bản
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)

        # Mobility
        player_moves = get_valid_moves(board, player)
        opponent_moves = get_valid_moves(board, opponent)
        mobility_score = len(player_moves) - len(opponent_moves)

        # Corner control
        corner_score = self.corner_heuristic(board, player) - self.corner_heuristic(board, opponent)

        # Edge stability
        stability_score = self.stability_heuristic(board, player) - self.stability_heuristic(board, opponent)

        # Parity (chẵn lẻ)
        parity_score = self.parity_heuristic(board)

        # Trả về điểm số với trọng số
        return (
            self.weights["score"] * (player_score - opponent_score) +
            self.weights["mobility"] * mobility_score +
            self.weights["corner"] * corner_score +
            self.weights["stability"] * stability_score +
            self.weights["parity"] * parity_score
        )

    def corner_heuristic(self, board, player):
        # Tính số góc mà người chơi kiểm soát
        return sum(1 for (r, c) in self.corner_positions if board[r, c] == player)

    def stability_heuristic(self, board, player):
        # Tính số quân cờ ổn định ở cạnh
        return sum(1 for (r, c) in self.edge_positions if board[r, c] == player and self.is_stable(board, r, c))

    def is_stable(self, board, r, c):
        # Kiểm tra xem quân cờ ở vị trí (r, c) có ổn định không
        # Quân cờ ổn định nếu nó nằm ở cạnh và không thể bị lật lại
        return (
            (r == 0 or r == 7 or c == 0 or c == 7) and
            (board[r, c] != 0)  # Không phải ô trống
        )

    def parity_heuristic(self, board):
        # Tính chẵn lẻ: số ô trống còn lại
        empty_cells = np.sum(board == 0)
        return 1 if empty_cells % 2 == 0 else -1

    def score_move(self, board, move, player, execute_move_fn):
        from helpers import get_valid_moves  # Import trong hàm để tránh circular import

        # Kiểm tra tính hợp lệ của nước đi
        if move not in get_valid_moves(board, player):
            return -float("inf")  # Invalid move = very bad score

        # Tạo bản sao bảng để thử nghiệm nước đi
        board_copy = board.copy()
        execute_move_fn(board_copy, move, player)

        # Đánh giá bảng sau khi thực hiện nước đi
        opponent = 3 - player
        return self.evaluate_board(board_copy, player, opponent, get_valid_moves)