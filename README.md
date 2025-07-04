# Reversi/Othello!

<p align="center">
  <img src="https://t4.ftcdn.net/jpg/00/90/53/03/240_F_90530312_4Mg3HCsCMW91NVHKWNlBaRo8F5pHhN3c.jpg?w=690&h=388&c=crop">
</p>

## Setup

To setup the game, clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Playing a game

To start playing a game, we will run the simulator and specify which agents should complete against eachother. To start, several agents are given to you, and you will add your own following the same game interface. For example, to play the game using two copies of the provided random agent (which takes a random action every turn), run the following:

```bash
python simulator.py --player_1 random_agent --player_2 random_agent
```

This will spawn a random game board of size NxN, and run the two agents of class [RandomAgent](agents/random_agent.py). You will be able to see their moves in the console.

## Visualizing a game

To visualize the moves within a game, use the `--display` flag. You can set the delay (in seconds) using `--display_delay` argument to better visualize the steps the agents take to win a game.

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --display
```

## Play on your own!

To take control of one side of the game and compete against the random agent yourself, use a [`human_agent`](agents/human_agent.py) to play the game.

```bash
python simulator.py --player_1 human_agent --player_2 random_agent --display
```

## Autoplaying multiple games

There is some randomness (coming from the initial game setup and potentially agent logic), so to fairly evaluate agents, we will run them against eachother multiple times, alternating their roles as player_1 and player_2, on various board sizes that are selected randomly (between size 6 and 12). The aggregate win % will determine a fair winner. Use the `--autoplay` flag to run $n$ games sequentially, where $n$ can be set using `--autoplay_runs`.

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --autoplay
```

During autoplay, boards are drawn randomly between size `--board_size_min` and `--board_size_max` for each iteration. You may try various ranges for your own information and development by providing these variables on the command-line. However, the defaults (to be used during grading) are 6 and 12, so ensure the timing limits are satisfied for every board in this size range. 

**Notes**

- Not all agents support autoplay (e.g. the human agent doesn't make sense this way). The variable `self.autoplay` in [Agent](agents/agent.py) can be set to `True` to allow the agent to be autoplayed. Typically this flag is set to false for a `human_agent`.
- UI display will be disabled in an autoplay.

## Develop your own general agent(s):

You need to write one agent and submit it for the class project, but you may develop additional agents during the development process to play against eachother, gather data or similar. To write a general agent:

1. Modify **ONLY** the [`student_agent.py`](agents/student_agent.py) file in [`agents/`](agents/) directory, which extends the [`agents.Agent`](agents/agent.py) class.
2. Do not add any additional imports.
3. Implement the `step` function with your game logic. Make extensive use of the functions imported from helpers.py which should be the majority of what you need to interact with the game. Any further logic can be coded directly in your file as global or class variables, functions, etc. Do not import world.py.
4. Test your performance against the random_agent with ```bash
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay```
5. Try playing against your own bot as a human. Consistently beating your own best-effort human play is a very good indicator of an A performance grade.

## Advanced and optional: What if I want to create other agents and test them against eachother?

There can only be one file called student_agent.py, and that's already perfectly set up to interact with our evaluation code, but you may create other agents during development. To get new files interacting correctly, you need to change a few specific things. Let's suppose you want to create second_agent.py, a second try at your student agent.

1. Create the new file by starting from a copy of the provided student_agent. ```$ cp agents/student_agent.py agents/second_agent.py```
2. Change the name in the decorator. Edit (@register_agent("student_agent")) instead to @register_agent("second_agent"), and the class name from `StudentAgent` to `SecondAgent`. 
3. Import your new agent in the [`__init__.py`](agents/__init__.py) file in [`agents/`](agents/) directory, by adding the line `from .second_agent import SecondAgent`
4. Now you can pit your two agents against each other in the simulator.py by running ```bash python simulator.py --player_1 student_agent --player_2 second_agent --display``` and see which idea is working better.
5. Adapt all of the above to create even more agents

## Full API

```bash
python simulator.py -h       
usage: simulator.py [-h] [--player_1 PLAYER_1] [--player_2 PLAYER_2]
                    [--board_size BOARD_SIZE] [--display]
                    [--display_delay DISPLAY_DELAY]

optional arguments:
  -h, --help            show this help message and exit
  --player_1 PLAYER_1
  --player_2 PLAYER_2
  --board_size BOARD_SIZE
  --display
  --display_delay DISPLAY_DELAY
  --autoplay
  --autoplay_runs AUTOPLAY_RUNS
```

## License

[MIT](LICENSE)

# Supervised Neural Network Agent

A new agent, `SupervisedAgent`, uses a neural network trained on historical Othello games from `othello_dataset.csv`.

## Training the Agent

1. Ensure `torch` is installed (see `requirements.txt`).
2. Run the training script:

```bash
python train_supervised_agent.py
```

This will process the dataset and save a trained model (e.g., `supervised_model.pth`).

## Using the Agent

After training, the agent can be used in the simulator by specifying `supervised_agent` as the agent name.
