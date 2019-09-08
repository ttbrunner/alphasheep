# AlphaSheep - Reinforcement Learning for Schafkopf

## Current Status
#### Simulator
- Mostly implemented: can run all major game modes.
- Bidding phase is still missing
- Currently only simulates single games (not keeping tabs over multiple rounds)
- Consequently, there is no "money" counting right now.
- Main classes: GameController, GameState

#### GUI
- Very basic PyGame implementation
- Completely optional - can be attached via event subscriber
- Single-threaded: blocks after every move, so the user can step through a game
- Currently, the user can only observe the agents but not play themselves.
- Run *play_single_game.py* to observe a single game in the GUI.

#### Agents
- Are asked for a card to play, based on the following information:
    - Current cards in (own) hand
    - Cards in current trick (on the table, played other players)
- Are notified of the game result.
- RandomCardAgent: Baseline that plays a random card that does not violate the rules
- DQNAgent: First try at a super-basic RL learner.

#### DQNAgent
- Cookie-cutter DQN implementation modeled after various books and tutorials.
- Uses discounted rewards
- State: 1-hot(x32) encoding of own cards, and those on the table. Nothing else.
- Action: 1-hot(x32) encoding of cards to play. The action space is of course limited by the cards that the player has, and that are valid in the current game mode.
- For faster training: modified to only learn every * episodes

#### Training regimen
- *train_rl_agent.py*: Runs games in a loop, training the agent, and saves checkpoints regularly.
- Currently, DQNAgent plays against 3 RandomCardAgents
- The dealing procedure is rigged so that Player 0 is always dealt good cards for a Herz-Solo
- Consequently, **the game mode is always Herz-Solo**. This is a very straightforward game mode.
- Agent performance is evaluated by win rate over 1000 games. During training, this is calculated as a moving average.
- RamdomCardAgent has ~60% winrate in the Herz-Solo scenario, whereas DQNAgent seems to fluctuate between 62-73.

#### Observations
- Find notes and ideas on experiments in *doc/observations.md*.

## TODO
- GUI: Add more info (who is declaring, scores, n_tricks?)
- GUI: Add interface for agents to provide debug info (e.g. q-value for every card)
- GUI: Add GUIAgent so the user can play themselves :)
- DQNAgent: Add eps-decay (for exploration)
- DQNAgent: Experiment with discount... the task is episodic, so it should work without?
- DQNAgent: Add inference mode (no exploration)
- DQNAgent: Expand state representation (history of tricks, mapping of cards to players)
- Training: Add eval script that runs parallel to training and tests a single snapshot of the agent.
- Eval: Write an agent that is the "best out of 100 RandomCardAgents" that play the same game, so we can determine how "winnable" a game really is. Then the performance measure can be "%of winnable games won" instead of total.
- After all that is working: test other game modes, and add bidding to Agents


## Prior work (might come in handy):
- RL: https://github.com/clauszitzelsberger/Schafkopf_RL
- MCTS: https://github.com/MartinDupont/SchafkopfBot



## How to train remotely with Slurm/Singularity

1. Create a Singularity container:
   ```
   cd _cluster_runner
   sudo singularity build sng-alphasheep.img sng_buildfile
   ```
2. Copy the image to the _cluster_runner dir on the remote machine
3. Start training for as many experiments you like:
   ```
   ./_cluster_experiment.sh "experiments/{my_experiment}.yaml"
   ```