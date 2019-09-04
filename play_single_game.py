"""
Runs a single game with an interactive GUI. Use for debugging purposes, or just to see the agents play.
"""
import argparse
import logging
import os

from agents.dqn_agent import DQNAgent
from agents.rule_based_agent import RuleBasedAgent
from controller.dealing_behavior import DealWinnableHand
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameMode, GameContract
from game.game_state import Player

from gui.gui import Gui, UserQuitGameException
from agents.agents import RandomCardAgent
from gui.gui_agent import GUIAgent
from log_util import init_logging, get_class_logger, get_named_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-agent", type=str, choices=['rule', 'random', 'alphasau', 'user'], required=True)
    parser.add_argument("--alphasau-checkpoint", help="Checkpoint for AlphaSau, if --p0-agent=alphasau.", required=False)
    args = parser.parse_args()
    agent_choice = args.p0_agent
    as_checkpoint_path = args.alphasau_checkpoint
    if agent_choice == "alphasau" and not as_checkpoint_path:
        raise ValueError("Need to specify --alphasau_checkpoint if --p0_agent=alphasau.")

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.DEBUG)        # Log every single card.
    get_class_logger(Gui).setLevel(logging.DEBUG)                   # Log mouse clicks.

    if agent_choice == "alphasau":
        get_class_logger(DQNAgent).setLevel(logging.DEBUG)
        alphasau_agent = DQNAgent(0, training=False)
        alphasau_agent.load_weights(as_checkpoint_path)
        p0 = Player("0-AlphaSau", agent=alphasau_agent)
    elif agent_choice == "user":
        p0 = Player("0-User", agent=GUIAgent(0))
    elif agent_choice == "rule":
        p0 = Player("0-Hans", agent=RuleBasedAgent(0))
        get_class_logger(RuleBasedAgent).setLevel(logging.DEBUG)
    else:
        p0 = Player("0-Hans", agent=RandomCardAgent(0))

    players = [
        p0,
        Player("1-Zenzi", agent=RandomCardAgent(1)),
        Player("2-Franz", agent=RandomCardAgent(2)),
        Player("3-Andal", agent=RandomCardAgent(3))
    ]

    # # Deal fairly and allow agents to choose their game.
    # controller = GameController(players)

    # Rig the game so Player 0 has the cards to play a Herz-Solo.
    # Also, force them to play it.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    # The GUI initializes PyGame and registers on events provided by the controller - then returns control.
    # The controller then runs the game as usual and fires GameStateChanged events, which the GUI receives.
    #
    # Since everything is done synchronously, the GUI can block on every event (and wait for the user to click).
    # In this way, the GUI can be used to debug and watch a single game.
    logger.info("Starting GUI.")
    with Gui(controller.game_state) as gui:
        # Run an endless loop of single games.
        logger.info("Starting game loop...")
        try:
            while True:
                controller.run_game()
        except UserQuitGameException:
            logger.info("User quit game.")

    logger.info("Shutdown.")


if __name__ == '__main__':
    main()
