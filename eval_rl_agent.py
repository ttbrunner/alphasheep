
import argparse
import logging
import os
from collections import deque

from agents.agents import RandomCardAgent
from agents.dqn_agent import DQNAgent
from controller.dealing_behavior import DealWinnableHand
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameContract, GameMode
from game.game_state import Player
from log_util import init_logging, get_class_logger, get_named_logger
from timeit import default_timer as timer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Weights are loaded from this file.")
    args = parser.parse_args()
    weights_path = args.weights_path

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    alphasau_agent = DQNAgent()
    if weights_path is not None:
        if not os.path.exists(weights_path):
            logger.info('Weights file "{}" does not exist. Will create new file.'.format(weights_path))
        else:
            logger.info('Loading weights from "{}..."'.format(weights_path))
            alphasau_agent.load_weights(weights_path)

    players = [
        Player("0-AlphaSau", agent=alphasau_agent),
        Player("1-Zenzi", agent=RandomCardAgent()),
        Player("2-Franz", agent=RandomCardAgent()),
        Player("3-Andal", agent=RandomCardAgent())
    ]

    # Rig the game so Player 0 has the cards to play a Herz-Solo. Force them to play it.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    # Train virtually forever.
    n_episodes = 100000000

    # Calculate win% as simple moving average.
    win_rate = float('nan')
    n_won = 0
    sma_window_len = 1000
    won_deque = deque()

    save_every_s = 60

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

            # Save model checkpoint
            if weights_path is not None and timer() - time_last_save > save_every_s:
                alphasau_agent.save_weights(weights_path, overwrite=True)
                time_last_save = timer()
                logger.info('Saved weights to "{}".'.format(weights_path))

        winners = controller.run_game()
        won = winners[0]
        won_deque.append(won)
        if won:
            n_won += 1

    logger.info("Finished playing.")
    logger.info("Final win rate: {:.1%}".format(win_rate))


if __name__ == '__main__':
    main()
