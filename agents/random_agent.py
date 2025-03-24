import numpy as np

from agents.agent import Agent
from helpers import random_move
from store import register_agent

# Important: you should register your agent with a name
@register_agent("random_agent")
class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True

    def step(self, chess_board, player, opponent):
        return random_move(chess_board, player)
