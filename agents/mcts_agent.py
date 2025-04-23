from agents.agent import Agent
from store import register_agent
import numpy as np # type: ignore
from copy import deepcopy
import time
from helpers import execute_move, check_endgame, get_valid_moves, get_directions

class TreeNode:
    def __init__(self, board, turn=None, action=None, parent=None, heuristic=None):
        self.board = board  # trạng thái bàn cờ hiện tại
        self.turn = turn  # lượt của người chơi hiện tại
        self.parent = parent  # nút cha
        self.action = action  # hành động được thực hiện để đến nút này
        self.children = []  # danh sách các nút con
        self.visits = 0  # số lần nút này được truy cập
        self.score = 0  # điểm số của nút này (số lần thắng)
        self.heuristic = heuristic  # hàm heuristic để đánh giá trạng thái bàn cờ

    def fully_expanded(self):
        return len(self.children) == len(get_valid_moves(self.board, self.turn))

    def uct_score(self, total_simulations, c=1.0):
        if self.visits == 0: return float('inf')
        # Tính điểm số UCT
        # UCT = tỉ lệ thắng + yếu tố khám phá + điểm thưởng đã chuẩn hóa
        win_ratio = self.score / self.visits
        exploration = c * np.sqrt(np.log(total_simulations + 1) / (self.visits + 1))
        bonus = self.parent.heuristic(self.board, self.turn, -self.turn) if self.parent else self.heuristic(self.board, self.turn, -self.turn)
        normalized = bonus / 2618
        return win_ratio + exploration + normalized

@register_agent("mcts_agent")
class MCTSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "MCTSAgent"
        self.max_iter = 100  # số vòng lặp tối đa cho MCTS
        self.max_time = 1.95  # thời gian tối đa cho MCTS (giây)

    def copy_board(self, board):
        return np.copy(board)

    def step(self, board, turn, rival):
        t0 = time.time()
        move = self.run_mcts(board, turn, rival)
        print("Thời gian ra quyết định của Agent:", time.time() - t0)
        return move
        
    # Kiểm tra một quân cờ có ổn định hay không (không bị lật)
    # Một quân cờ ổn định nếu nằm ở rìa hoặc xung quanh là các quân cùng màu
    def stable(self, board, x, y, turn):
        if x in [0, len(board) - 1] or y in [0, len(board[0]) - 1]:
            return True
        for dx, dy in get_directions():
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and board[nx][ny] != turn:
                return False
        return True

    # Thực hiện một nước đi và trả về bàn cờ mới
    def apply_move(self, board, move, turn):
        new_board = deepcopy(board)
        execute_move(new_board, move, turn)
        return new_board

    # Chạy thuật toán MCTS
    def run_mcts(self, board, turn, rival):
        root = TreeNode(board=board, turn=turn, heuristic=self.evaluate)
        self.expand_node(root)
        sim_count = 0
        trials = 2
        depth = 5
        filled = np.count_nonzero(board)
        phase = filled / (len(board) * len(board))
        randomness = 0.1 if phase < 0.5 else 0.05
        node = self.select_node(root, sim_count)
        w, l, d = self.simulate_game(node, sims=trials, limit=depth, eps=randomness)
        self.propagate(node, w, l, d)
        sim_count += trials * 2
        i = 0
        t_start = time.time()

        while time.time() - t_start < self.max_time and i < self.max_iter:
            node = self.select_node(root, sim_count)
            if not check_endgame(node.board, node.turn, -node.turn)[0]:
                self.expand_node(node)
            w, l, d = self.simulate_game(node, sims=trials, limit=depth, eps=randomness)
            self.propagate(node, w, l, d)
            sim_count += trials * 2
            i += 1

        best = max(root.children, key=lambda child: child.visits)
        return best.action

    # Chọn node tiếp theo để duyệt dựa trên thuật toán UCT
    def select_node(self, node, total):
        while node.children and node.fully_expanded():
            node = max(node.children, key=lambda child: child.uct_score(total))
        return node

    # Mở rộng node hiện tại bằng cách thêm các node con cho mỗi nước đi hợp lệ
    def expand_node(self, node):
        moves = get_valid_moves(node.board, node.turn)
        used = {child.action for child in node.children}
        for m in sorted(moves, key=lambda mv: self.evaluate(self.do(node.board, mv, node.turn), node.turn, -node.turn), reverse=True):
            if m not in used:
                new_state = deepcopy(node.board)
                execute_move(new_state, m, node.turn)
                child_node = TreeNode(board=new_state, turn=-node.turn, action=m, parent=node, heuristic=self.evaluate)
                node.children.append(child_node)
                break

    # Mô phỏng một ván chơi từ node hiện tại đến khi kết thúc (thắng/thua/hòa)
    def simulate_game(self, node, sims=5, limit=10, eps=0.1):
        board_copy = self.copy_board(node.board)
        current = node.turn
        wins = losses = draws = 0

        for _ in range(sims):
            sim_board = self.copy_board(board_copy)
            turn = current
            for _ in range(limit):
                game_over, s1, s2 = check_endgame(sim_board, turn, -turn)
                if game_over:
                    if s1 > s2:
                        wins += 1
                    elif s1 < s2:
                        losses += 1
                    else:
                        draws += 1
                    break
                options = get_valid_moves(sim_board, turn)
                if not options:
                    turn = -turn
                    continue
                pick = options[np.random.randint(len(options))] if np.random.rand() < eps else sorted(
                    options, key=lambda mv: self.evaluate(self.do(node.board, mv, node.turn), node.turn, -node.turn), reverse=True
                )[0]
                execute_move(sim_board, pick, turn)
                turn = -turn

        return wins, losses, draws

    # Lan truyền kết quả mô phỏng lên các node tổ tiên trong cây
    def propagate(self, node, win, loss, draw):
        side = node.turn
        while node:
            node.visits += win + loss + draw
            if node.turn == side:
                node.score += win
            else:
                node.score += loss
            node = node.parent

    # Đánh giá trạng thái bàn cờ bằng hàm heuristic
    def evaluate(self, board, p, o):
        c_bonus = 100
        center_area = [(3, 3), (3, 4), (4, 3), (4, 4)]
        total = board.shape[0] * board.shape[1]
        occupied = np.sum(board != 0)
        stage = occupied / total
        score_corner = score_stable = score_move = score_center = score_piece = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        # Tính điểm bonus cho các góc
        for x, y in corners:
            if board[x][y] == p:
                score_corner += c_bonus
            elif board[x][y] == o:
                score_corner -= c_bonus
            if self.stable(board, x, y, p):
                score_corner += c_bonus * 2

        for cx, cy in center_area:
            if board[cx][cy] == p:
                score_center += 10

        # Tính điểm bonus cho các quân cờ ổn định
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == p and self.stable(board, i, j, p):
                    score_stable += 10
                elif board[i][j] == o and self.stable(board, i, j, o):
                    score_stable -= 10
                if i in [0, 7] or j in [0, 7]:
                    score_stable += 5

        # Tính điểm cho số nước đi hợp lệ
        mp = len(get_valid_moves(board, p))
        mo = len(get_valid_moves(board, o))

        for mv in get_valid_moves(board, p):
            next_board = self.apply_move(board, mv, p)
            future_mp = len(get_valid_moves(next_board, p))
            if future_mp < mp:
                score_move -= 20

        # Thay đổi cách đánh giá điểm cho số nước đi hợp lệ theo giai đoạn
        # Giai đoạn đầu: 20 điểm cho mỗi nước đi hợp lệ
        # Giai đoạn giữa: 10 điểm cho mỗi nước đi hợp lệ
        # Giai đoạn cuối: 5 điểm cho mỗi nước đi hợp lệ
        if stage < 0.3:
            score_move = 20 * (mp - mo)
        elif stage < 0.7:
            score_move = 10 * (mp - mo)
        else:
            score_move = 5 * (mp - mo)

        # Tính điểm cho số quân cờ của mỗi người chơi
        count_p = np.sum(board == p)
        count_o = np.sum(board == o)
        if stage < 0.3:
            score_piece = 0.1 * (count_p - count_o)
        elif stage < 0.7:
            score_piece = 0.5 * (count_p - count_o)
        else:
            score_piece = 1.0 * (count_p - count_o)

        return 0.5 * score_move + 1.5 * score_corner + 0.5 * score_stable + 0.3 * score_center + score_piece

    # Thực hiện nước đi và trả về trạng thái bàn cờ mới
    def do(self, board, mv, turn):
        b = self.copy_board(board)
        execute_move(b, mv, turn)
        return b
