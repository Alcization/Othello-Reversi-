 # type: ignore
from world import World, PLAYER_1_NAME, PLAYER_2_NAME
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from utils import all_logging_disabled
import logging
import numpy as np
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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

def simulate_single_match(args_dict):
    import numpy as np
    from simulator import Simulator
    from utils import all_logging_disabled

    sim = Simulator(args_dict['args'])
    sim.args.display = False
    sim.valid_board_sizes = args_dict['valid_board_sizes']
    swap_players = args_dict['swap']
    board_size = np.random.choice(sim.valid_board_sizes)

    with suppress_stdout(), all_logging_disabled():
        p0_score, p1_score, p0_time, p1_time = sim.run(
            swap_players=swap_players, board_size=board_size
        )

    if swap_players:
        p0_score, p1_score, p0_time, p1_time = p1_score, p0_score, p1_time, p0_time
    return p0_score, p1_score, p0_time, p1_time

class Simulator:
    def __init__(self, args):
        self.args = args
        self.valid_board_sizes = [ i for i in range(self.args.board_size_min, self.args.board_size_max+1) if i % 2 == 0 ]

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
        from copy import deepcopy

        self.args.display = False
        tasks = []
        results = []
        with all_logging_disabled(), ProcessPoolExecutor() as executor:
            for i in range(self.args.autoplay_runs):
                swap_players = i % 2 == 0
                args_copy = deepcopy(self.args)
                tasks.append(executor.submit(simulate_single_match, {
                    'args': args_copy,
                    'swap': swap_players,
                    'valid_board_sizes': self.valid_board_sizes,
                }))

            for i, future in enumerate(as_completed(tasks), 1):
                print(f"Complete match {i}/{self.args.autoplay_runs}...")
                results.append(future.result())

        p1_win_count = 0
        p2_win_count = 0
        p1_times = []
        p2_times = []
        for p0_score, p1_score, p0_time, p1_time in results:
            if p0_score > p1_score:
                p1_win_count += 1
            elif p0_score < p1_score:
                p2_win_count += 1
            else:
                p1_win_count += 0.5
                p2_win_count += 0.5
            p1_times.extend(p0_time)
            p2_times.extend(p1_time)

        logger.info(f"Complete: {self.args.autoplay_runs} match(s).")
        logger.info(
            f"Player 1, agent {self.args.player_1}, win count: {p1_win_count}, win percentage: {p1_win_count / self.args.autoplay_runs}. Maximum turn time was {np.max(p1_times):.5f} s. Minimum: {np.min(p1_times):.5f}, Average: {np.mean(p1_times):.5f}"
        )
        logger.info(
            f"Player 2, agent {self.args.player_2}, win count: {p2_win_count}, win percentage: {p2_win_count / self.args.autoplay_runs}. Maximum turn time was {np.max(p2_times):.5f} s. Minimum: {np.min(p2_times):.5f}, Average: {np.mean(p2_times):.5f}"
        )

if __name__ == "__main__":
    args = get_args()
    simulator = Simulator(args)
    if args.autoplay:
        simulator.autoplay()
    else:
        simulator.run()
