# Human Input agent
import sys

from agents.agent import Agent
from store import register_agent


@register_agent("human_agent")
class HumanAgent(Agent):
    def __init__(self):
        super(HumanAgent, self).__init__()
        self.name = "HumanAgent"

    def step(self, chess_board, player, opponent):
        text = input("Your move (row,column) or input q to quit: ")
        while len(text.split(",")) != 2 and "q" not in text.lower():
            print("Wrong Input Format! Input should be row,column.")
            text = input("Your move (row,column) or input q to quit: ")

        if "q" in text.lower():
            print("Game ended by user!")
            sys.exit(0)

        x, y = text.split(",")
        x, y = int(x.strip()), int(y.strip())

        while not self.check_valid_input(x, y, chess_board):
            print(
                "Invalid Move! (row,column) should be within the board and the position must be empty."
            )
            text = input("Your move (row,column) or input q to quit: ")
            while len(text.split(",")) != 2 and "q" not in text.lower():
                print("Wrong Input Format! Input should be row,column.")
                text = input("Your move (row,column) or input q to quit: ")
            if "q" in text.lower():
                print("Game ended by user!")
                sys.exit(0)
            x, y = text.split(",")
            x, y = int(x.strip()), int(y.strip())

        return (x, y)

    def check_valid_input(self, x, y, chess_board):
        board_size = chess_board.shape[0]
        return 0 <= x < board_size and 0 <= y < board_size and chess_board[x, y] == 0
