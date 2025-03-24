import numpy as np

def get_directions():
    return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

def count_capture(chess_board, move_pos, player):
    r, c = move_pos
    if chess_board[r, c] != 0:
        return 0
    
    captured = 0

    # Check if move captures any opponent discs in any direction
    for dir in get_directions():
        captured = captured + count_capture_dir(chess_board,move_pos, player, dir)

    return captured

def count_capture_dir(chess_board, move_pos, player, direction):
    r, c = move_pos
    dx, dy = direction
    r += dx
    c += dy
    captured = 0
    board_size = chess_board.shape[0]

    while 0 <= r < board_size and 0 <= c < board_size:
        if chess_board[r, c] == 0:
            return 0
        if chess_board[r, c] == player:
            return captured
        captured = captured + 1
        r += dx
        c += dy

    return 0


def execute_move(chess_board, move_pos, player):
    r, c = move_pos
    chess_board[r, c] = player

    # Flip opponent's discs in all directions where captures occur
    for direction in get_directions():
        flip_discs(chess_board,move_pos, player, direction)

def flip_discs(chess_board, move_pos, player, direction):
    
    if count_capture_dir(chess_board,move_pos, player, direction) == 0:
        return
    
    r, c = move_pos
    dx, dy = direction
    r += dx
    c += dy

    while chess_board[r, c] != player:
        chess_board[r, c] = player
        r += dx
        c += dy

def check_endgame(chess_board,player,opponent):
    is_endgame = False

    valid_moves = get_valid_moves(chess_board,player)
    if not valid_moves:
        opponent_valid_moves = get_valid_moves(chess_board,opponent)
        if not opponent_valid_moves:
            is_endgame = True  # When no-one can play, the game is over, score is current piece count

    p0_score = np.sum(chess_board == 1)
    p1_score = np.sum(chess_board == 2)
    return is_endgame, p0_score, p1_score

def get_valid_moves(chess_board,player):
    board_size = chess_board.shape[0]
    valid_moves = []
    for r in range(board_size):
        for c in range(board_size):
            if count_capture(chess_board,(r, c), player) > 0:
                valid_moves.append((r, c))

    return valid_moves

def random_move(chess_board, player):
    valid_moves = get_valid_moves(chess_board,player)

    if len(valid_moves) == 0:
        # If no valid moves are available, return None
        print(f"No valid moves left for player {player}.")
        return None
    
    return valid_moves[np.random.randint(len(valid_moves))]
