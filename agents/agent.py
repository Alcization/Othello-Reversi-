class Agent:
    def __init__(self):
        self.name = "DummyAgent"
        self.autoplay = True

    def __str__(self) -> str:
        return self.name

    def step(self, chess_board, player, opponent):
        pass
