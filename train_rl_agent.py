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
from timeit import default_timer as timer


def main():
    # Game Setup:
    # - Every game has the ego player (id 0) playing a Herz-Solo.
    # - The cards are rigged so that the ego player always receives a pretty good hand, most of them are winnable.
    # - The 3 enemies are all RandomCardAgents - for now.
    #
    # Observations so far:
    # - If the ego player is also a RandomCardAgent(), they have a 60.4% win rate (averaged over 100k games).
    # - The RL player should manage to beat that.
    #
    # Next steps:
    # - See what happens - can the DQN agent learn a win percentage that is significantly higher?
    # - Speed up learning

    players = [
        Player("0-AlphaSau", agent=DQNAgent()),
        Player("1-Zenzi", agent=RandomCardAgent()),
        Player("2-Franz", agent=RandomCardAgent()),
        Player("3-Andal", agent=RandomCardAgent())
    ]

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("train_rl_agent.main")
    get_class_logger(GameController).setLevel(logging.INFO)

    # Rig the game so Player 0 has the cards to play a Herz-Solo.
    # Also, force them to play it.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    n_episodes = 10000000
    n_won = 0
    time_start = timer()
    for i_episode in range(n_episodes):
        if i_episode % 100 == 0 and i_episode > 0:
            s_elapsed = timer() - time_start
            logger.info("Ran {} Episodes. Win rate is {:.1%}. Speed is {:.0f} episodes/second.".format(
                i_episode, n_won/i_episode, i_episode/s_elapsed))

        winners = controller.run_game()
        if winners[0]:
            n_won += 1


    logger.info("Finished playing.")
    logger.info("Total win rate: {:.1%}".format(n_won/n_episodes))


if __name__ == '__main__':
    main()
