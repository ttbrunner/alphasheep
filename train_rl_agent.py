"""
Runs a large number of games without the GUI. Use this to train an agent.
"""
import argparse
import logging
import os
import shutil
from collections import deque

from agents.dummy.random_card_agent import RandomCardAgent
from agents.reinforcment_learning.dqn_agent import DQNAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from controller.dealing_behavior import DealWinnableHand
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameContract, GameMode
from game.game_state import Player
from utils.log_util import init_logging, get_class_logger, get_named_logger
from timeit import default_timer as timer

from utils.config_util import load_config


def main():
    # Game Setup:
    # - Every game has the ego player (id 0) playing a Herz-Solo.
    # - The cards are rigged so that the ego player always receives a pretty good hand, most of them are winnable.

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="A yaml config file. Must always be specified.", required=True)
    args = parser.parse_args()

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    # Load config.
    # Create experiment dir and prepend it to all paths.
    # If it already exists, then training will simply resume.
    logger.info(f'Loading config from "{args.config}"...')
    config = load_config(args.config)
    experiment_dir = config["experiment_dir"]
    os.makedirs(config["experiment_dir"], exist_ok=True)
    agent_checkpoint_paths = {i: os.path.join(experiment_dir, name) for i, name in config["training"]["agent_checkpoint_names"].items()}

    # Create agents.
    agents = []
    for i in range(4):
        x = config["training"]["player_agents"][i]
        if x == "DQNAgent":
            agent = DQNAgent(i, config=config, training=True)
        elif x == "RandomCardAgent":
            agent = RandomCardAgent(i)
        elif x == "RuleBasedAgent":
            agent = RuleBasedAgent(i)
        else:
            raise ValueError(f'Unknown agent type: "{x}"')
        agents.append(agent)

    # Load weights for agents.
    for i, weights_path in agent_checkpoint_paths.items():
        if not os.path.exists(weights_path):
            logger.info('Weights file "{}" does not exist. Will create new file.'.format(weights_path))
        else:
            agents[i].load_weights(weights_path)

    players = [Player(f"Player {i} ({a.__class__.__name__})", agent=a) for i, a in enumerate(agents)]

    # Rig the game so Player 0 has the cards to play a Herz-Solo. Force them to play it.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    n_episodes = config["training"]["n_episodes"]
    logger.info(f"Will train for {n_episodes} episodes.")

    # Calculate win% as simple moving average.
    win_rate = float('nan')
    n_won = 0
    sma_window_len = 1000
    won_deque = deque()

    save_every_s = config["training"]["save_checkpoints_every_s"]

    time_start = timer()
    time_last_save = timer()
    for i_episode in range(n_episodes):

        if i_episode > 0:
            # Calculate avg win%
            if i_episode < sma_window_len:
                win_rate = n_won / i_episode
            else:
                if won_deque.popleft() is True:
                    n_won -= 1
                win_rate = n_won / sma_window_len

            # Log
            if i_episode % 100 == 0:
                s_elapsed = timer() - time_start
                logger.info("Ran {} Episodes. Win rate (last {} episodes) is {:.1%}. Speed is {:.0f} episodes/second.".format(
                    i_episode, sma_window_len, win_rate, i_episode/s_elapsed))

            # Save model checkpoint.
            # Also make a copy for evaluation - the eval jobs will sync on this file and later remove it.
            if timer() - time_last_save > save_every_s:
                for i, weights_path in agent_checkpoint_paths.items():
                    agents[i].save_weights(weights_path, overwrite=True)
                    shutil.copyfile(weights_path, f"{os.path.splitext(weights_path)[0]}.for_eval.h5")
                time_last_save = timer()

        winners = controller.run_game()
        won = winners[0]
        won_deque.append(won)
        if won:
            n_won += 1

    logger.info("Finished playing.")
    logger.info("Final win rate: {:.1%}".format(win_rate))


if __name__ == '__main__':
    main()
