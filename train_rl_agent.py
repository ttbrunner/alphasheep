"""
Runs a large number of games without the GUI. Use this to train an agent.
"""
import logging

from agents.agents import RandomCardAgent
from agents.dqn_agent import DQNAgent
from controller.dealing_behavior import DealWinnableHand
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameContract, GameMode
from game.game_state import Player
from log_util import init_logging, get_class_logger, get_named_logger


def main():
    # For starters, run a single game with a RL mock agent.
    players = [
        Player("0-AlphaSau", agent=RandomCardAgent()),
        Player("1-Zenzi", agent=RandomCardAgent()),
        Player("2-Franz", agent=DQNAgent()),
        Player("3-Andal", agent=RandomCardAgent())
    ]

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("train_rl_agent.main")
    get_class_logger(GameController).setLevel(logging.DEBUG)

    # Rig the game so Player 0 has the cards to play a Herz-Solo.
    # Also, force them to play it.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    n_episodes = 50
    for i_episode in range(n_episodes):
        logger.info("Running episode (game) {}...".format(i_episode))
        controller.run_game()


if __name__ == '__main__':
    main()
