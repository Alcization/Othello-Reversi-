
import numpy as np # type: ignore

from helpers import execute_move, check_endgame, get_valid_moves

def evaluate_player_current_score(chess_board: np.ndarray, player: int) -> float:
    """
    Evaluate the current score for a player.
    The score is calculated as the difference between the number of disks for the player and the opponent.
    """
    player_score = np.sum(chess_board == player)
    return player_score


def evaluate_mobility(chess_board: np.ndarray, player: int) -> float:
    """
    Evaluate the mobility score for a player.
    The mobility score is calculated as the difference between the number of valid moves for the player and the opponent.
    """
    player_moves = len(get_valid_moves(chess_board, player))
    return player_moves


# helper function of check_stable_disk
def count_corner_stable_disks(chess_board: np.ndarray, player: int, loop: int) -> int:
    """
    Count the number of corner stable disks of each INNER loop.
    """

    def get_adjacent_corners(chess_board: np.ndarray, player: int, loop: int, list_corners: list[tuple[int, int]], position: int) -> list[tuple[int, int]]:
        """
        Get the adjacent corners of a disk of inner loop (from loop = 1)
        """
        size = chess_board.shape[0]

        # position = 0: top-left corner
        # position = 1: top-right corner
        # position = 2: bottom-left corner
        # position = 3: bottom-right corner

        row, col = list_corners[position]

        if (chess_board[list_corners[0]] == player) and (position == 0 and loop != 0):
                return [(row - 1, col), (row, col - 1), (row - 1, col - 1)]
        elif (chess_board[list_corners[1]] == player) and (position == 1 and loop != 0):
            return [(row - 1, col), (row, col + 1), (row - 1, col + 1)]
        elif (chess_board[list_corners[2]] == player) and (position == 2 and loop != 0):
            return [(row + 1, col), (row, col - 1), (row + 1, col - 1)]
        elif (chess_board[list_corners[3]] == player) and (position == 3 and loop != 0):
            return [(row + 1, col), (row, col + 1), (row + 1, col + 1)]
        else:
            return []

    size = chess_board.shape[0]

    # Order of corner_in_loop: top left, top right, bottom left, bottom right
    corner_in_loop: list[tuple[int, int]] = [(loop, loop), (loop, size - loop - 1), (size - loop - 1, loop),
                                             (size - loop - 1, size - loop - 1)]

    corner_stable_disks: int = 0
    corner_position: int = 0

    for corner in corner_in_loop:
        if chess_board[corner] == player and loop == 0:
            corner_stable_disks += 1
        elif chess_board[corner] == player and loop != 0:
            # Check the adjacent corners of the disk
            adjacent_corners: list[tuple[int, int]] = get_adjacent_corners(chess_board, player, loop, corner_in_loop, corner_position)
            corner_position += 1
            is_corner_stable: bool = True
            for adjacent_corner in adjacent_corners:
                if chess_board[adjacent_corner] != player:
                    is_corner_stable = False
                    break
            if is_corner_stable:
                corner_stable_disks += 1
    return corner_stable_disks

# helper function of check_stable_disk
def evaluate_stable_disks(chess_board: np.ndarray, player: int) -> float:
    """
    Count the number of stable disks of player on Othello board
    """

    size: int = chess_board.shape[0]

    # Count the corners in the OUTER LOOP of player and opponent
    corners: list[tuple[int, int]] = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]

    player_corner: int = count_corner_stable_disks(chess_board, player, 0)

    # If no corners in the OUTER LOOP, it means there are no stable disks for both players
    if player_corner == 0:
        return 0

    def evaluate_outer_edge_stable_disks(chess_board: np.ndarray, player: int, loop: int, edge_position: str) -> int:
        """
        Check whether a disk at the edge of the board is stable.
        """

        def check_outer_adjacent_square(disk: tuple[int, int], chess_board: np.ndarray, player: int, loop: int,
                                        position: str) -> bool:
            if loop != 0:
                row, col = disk
                if position == "top" and chess_board[row + 1, col] == player:
                    return True
                elif position == "right" and chess_board[row, col - 1] == player:
                    return True
                elif position == "bottom" and chess_board[row - 1, col] == player:
                    return True
                elif position == "left" and chess_board[row, col + 1] == player:
                    return True
            return False

        size = chess_board.shape[0]
        corner_in_loop = [(loop, loop), (loop, size - loop - 1), (size - loop - 1, loop),
                          (size - loop - 1, size - loop - 1)]
        stable_disk = 0

        if edge_position == "top":
            for i in range(1 + loop, size - loop - 1):
                if ((chess_board[loop, i] == player) and loop == 0) or (
                        loop != 0 and check_outer_adjacent_square((loop, i), chess_board, player, loop, "top")):
                    if not (((loop, i) in corner_in_loop) and (loop == 0)):
                        stable_disk += 1
                else:
                    break
            if stable_disk == size - (loop + 1) * 2:
                return stable_disk
            else:
                for i in reversed(range(1 + loop, size - loop - 1)):
                    if ((chess_board[loop, i] == player) and loop == 0) or (
                            loop != 0 and check_outer_adjacent_square((loop, i), chess_board, player, loop, "top")):
                        stable_disk += 1
                    else:
                        break
        elif edge_position == "right":
            for i in range(1 + loop, size - loop - 1):
                if ((chess_board[i, size - loop - 1] == player) and loop == 0) or (
                        loop != 0 and check_outer_adjacent_square((i, size - loop - 1), chess_board, player, loop,
                                                                  "right")):
                    if not (((i, size - loop - 1) in corner_in_loop) and (loop == 0)):
                        stable_disk += 1
                else:
                    break
            if stable_disk == size - (loop + 1) * 2:
                return stable_disk
            else:
                for i in reversed(range(1 + loop, size - loop - 1)):
                    if ((chess_board[i, size - loop - 1] == player) and loop == 0) or (
                            loop != 0 and check_outer_adjacent_square((i, size - loop - 1), chess_board, player, loop,
                                                                      "right")):
                        stable_disk += 1
                    else:
                        break
        elif edge_position == "bottom":
            for i in range(1 + loop, size - loop - 1):
                if ((chess_board[size - loop - 1, i] == player) and loop == 0) or (
                        loop != 0 and check_outer_adjacent_square((size - loop - 1, i), chess_board, player, loop,
                                                                  "bottom")):
                    if not (((size - loop - 1, i) in corner_in_loop) and (loop == 0)):
                        stable_disk += 1
                else:
                    break
            if stable_disk == size - (loop + 1) * 2:
                return stable_disk
            else:
                for i in reversed(range(1 + loop, size - loop - 1)):
                    if ((chess_board[size - loop - 1, i] == player) and loop == 0) or (
                            loop != 0 and check_outer_adjacent_square((size - loop - 1, i), chess_board, player, loop,
                                                                      "bottom")):
                        stable_disk += 1
                    else:
                        break
        elif edge_position == "left":
            for i in range(1 + loop, size - loop - 1):
                if ((chess_board[i, loop] == player) and loop == 0) or (
                        loop != 0 and check_outer_adjacent_square((i, loop), chess_board, player, loop, "left")):
                    if not (((i, loop) in corner_in_loop) and (loop == 0)):
                        stable_disk += 1
                else:
                    break
            if stable_disk == size - (loop + 1) * 2:
                return stable_disk
            else:
                for i in reversed(range(1 + loop, size - loop - 1)):
                    if ((chess_board[i, loop] == player) and loop == 0) or (
                            loop != 0 and check_outer_adjacent_square((i, loop), chess_board, player, loop, "left")):
                        stable_disk += 1
                    else:
                        break

        return stable_disk

    # The number of inner circle (or loops) inside the othello board
    number_of_loop: int = int(size / 2)

    # Most outer loop
    stable_disk: int = \
        evaluate_outer_edge_stable_disks(chess_board, player, 0, "top") + \
        evaluate_outer_edge_stable_disks(chess_board, player, 0, "right") + \
        evaluate_outer_edge_stable_disks(chess_board, player, 0, "bottom") + \
        evaluate_outer_edge_stable_disks(chess_board, player, 0, "left")

    # Inner loop
    inner_loop_stable_disk: int
    for loop in range(1, number_of_loop - 1):
        inner_loop_stable_disk = 0
        # Count number of corner stable disks
        inner_loop_stable_disk += count_corner_stable_disks(chess_board, player, loop)

        # No corners mean no stable disks
        if inner_loop_stable_disk == 0:
            break

        # Count number of stable disks on the edge

        inner_loop_stable_disk += (evaluate_outer_edge_stable_disks(chess_board, player, loop, "top") +
                                   evaluate_outer_edge_stable_disks(chess_board, player, loop, "right") +
                                   evaluate_outer_edge_stable_disks(chess_board, player, loop, "bottom") +
                                   evaluate_outer_edge_stable_disks(chess_board, player, loop, "left"))
        if inner_loop_stable_disk == 0:
            break
        stable_disk += inner_loop_stable_disk

    return stable_disk

def evaluate_score_comparison(player_score: float, opponent_score: float) -> float:
    """
    Compare the scores of two players and return the difference.
    A positive score indicates the player is ahead, while a negative score indicates they are behind.
    """
    return (player_score - opponent_score) / (player_score + opponent_score + 1e-6) * 100
