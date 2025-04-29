# type: ignore
from world import World, PLAYER_1_NAME, PLAYER_2_NAME
import argparse
from utils import all_logging_disabled
import logging
import numpy as np
import datetime

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_1", type=str, default="random_agent")
    parser.add_argument("--player_2", type=str, default="random_agent")
    parser.add_argument("--board_size", type=int, default=None)
    parser.add_argument(
        "--board_size_min",
        type=int,
        default=6,
        help="In autoplay mode, the minimum board size",
    )
    parser.add_argument(
        "--board_size_max",
        type=int,
        default=12,
        help="In autoplay mode, the maximum board size",
    )
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_delay", type=float, default=0.3)
    parser.add_argument("--display_save", action="store_true", default=False)
    parser.add_argument("--display_save_path", type=str, default="plots/")
    parser.add_argument("--autoplay", action="store_true", default=False)
    parser.add_argument("--autoplay_runs", type=int, default=100)
    args = parser.parse_args()
    return args


class Simulator:
    def __init__(self, args):
        self.args = args
        # Only play on even-sized boards
        self.valid_board_sizes = [i for i in range(self.args.board_size_min, self.args.board_size_max + 1) if
                                  i % 2 == 0]
        # print("Valid sizes: ",self.valid_board_sizes)

    def reset(self, swap_players=False, board_size=None):
        if board_size is None:
            board_size = self.args.board_size
        if swap_players:
            player_1, player_2 = self.args.player_2, self.args.player_1
        else:
            player_1, player_2 = self.args.player_1, self.args.player_2

        self.world = World(
            player_1=player_1,
            player_2=player_2,
            board_size=board_size,
            display_ui=self.args.display,
            display_delay=self.args.display_delay,
            display_save=self.args.display_save,
            display_save_path=self.args.display_save_path,
            autoplay=self.args.autoplay,
        )

    def run(self, swap_players=False, board_size=None):
        self.reset(swap_players=swap_players, board_size=board_size)
        is_end, p0_score, p1_score = self.world.step()
        while not is_end:
            is_end, p0_score, p1_score = self.world.step()
        logger.info(
            f"Run finished. {PLAYER_1_NAME} player, agent {self.args.player_1}: {p0_score}. {PLAYER_2_NAME}, agent {self.args.player_2}: {p1_score}"
        )
        return p0_score, p1_score, self.world.p0_time, self.world.p1_time

    def autoplay(self):
        p1_win_count = 0
        p2_win_count = 0
        p1_times = []
        p2_times = []
        if self.args.display:
            logger.warning("Since running autoplay mode, display will be disabled")
        self.args.display = False
        with all_logging_disabled():
            for i in range(self.args.autoplay_runs):
                print(f"Running match {i + 1}/{self.args.autoplay_runs}...")
                swap_players = i % 2 == 0
                board_size = self.valid_board_sizes[np.random.randint(len(self.valid_board_sizes))]
                p0_score, p1_score, p0_time, p1_time = self.run(
                    swap_players=swap_players, board_size=board_size
                )
                if swap_players:
                    p0_score, p1_score, p0_time, p1_time = (
                        p1_score,
                        p0_score,
                        p1_time,
                        p0_time,
                    )
                if p0_score > p1_score:
                    p1_win_count += 1
                elif p0_score < p1_score:
                    p2_win_count += 1
                else:  # Tie
                    p1_win_count += 0.5
                    p2_win_count += 0.5
                p1_times.extend(p0_time)
                p2_times.extend(p1_time)

        logger.info(
            f"Complete: {self.args.autoplay_runs} match(s)."
        )
        logger.info(
            f"Player 1, agent {self.args.player_1}, win count: {p1_win_count}, win percentage: {p1_win_count / self.args.autoplay_runs}. Maximum turn time was {np.round(np.max(p1_times), 5)} seconds. Minimum turn time was {np.round(np.min(p1_times), 5)} seconds. Average turn time was {np.round(np.mean(p1_times), 5)} seconds."
        )
        logger.info(
            f"Player 2, agent {self.args.player_2}, win count: {p2_win_count}, win percentage: {p2_win_count / self.args.autoplay_runs}. Maximum turn time was {np.round(np.max(p2_times), 5)} seconds. Minimum turn time was {np.round(np.min(p2_times), 5)} seconds. Average turn time was {np.round(np.mean(p2_times), 5)} seconds."
        )


if __name__ == "__main__":
    args = get_args()
    simulator = Simulator(args)
    if args.autoplay:
        simulator.autoplay()
    else:
        simulator.run()