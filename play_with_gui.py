"""
Runs games with an interactive GUI.

Use this for playing yourself, or watching other agents play.
Right now, only Player 0's agent can be specified, the others are RuleBasedAgents.
"""
import argparse
import logging
import os

from agents.reinforcment_learning.dqn_agent import DQNAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from agents.dummy.static_policy_agent import StaticPolicyAgent
from simulator.controller.dealing_behavior import DealWinnableHand
from simulator.controller.game_controller import GameController
from simulator.card_defs import Suit
from simulator.game_mode import GameMode, GameContract
from simulator.game_state import Player

from gui.gui import Gui, UserQuitGameException
from agents.dummy.random_card_agent import RandomCardAgent
from gui.gui_agent import GUIAgent
from utils.log_util import init_logging, get_class_logger, get_named_logger
from utils.config_util import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-agent", type=str, choices=['static', 'rule', 'random', 'alphasau', 'user'], required=True)
    parser.add_argument("--alphasau-checkpoint", help="Checkpoint for AlphaSau, if --p0-agent=alphasau.", required=False)
    parser.add_argument("--agent-config", help="YAML file, containing agent specifications for AlphaSau.", required=False)
    args = parser.parse_args()
    agent_choice = args.p0_agent
    as_checkpoint_path = args.alphasau_checkpoint
    as_config_path = args.agent_config
    if agent_choice == "alphasau" and (not as_checkpoint_path or not as_config_path):
        raise ValueError("Need to specify --alphasau-checkpoint and --agent-config if --p0_agent=alphasau.")

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.DEBUG)        # Log every single card.
    get_class_logger(Gui).setLevel(logging.DEBUG)                   # Log mouse clicks.
    get_class_logger(RuleBasedAgent).setLevel(logging.DEBUG)        # Log decisions by the rule-based players.

    # Create the agent for Player 0.
    if agent_choice == "alphasau":

        # Load config. We ignore the "training" and "experiment" sections, but we need "agent_config".
        logger.info(f'Loading config from "{as_config_path}"...')
        config = load_config(as_config_path)
        get_class_logger(DQNAgent).setLevel(logging.DEBUG)          # Log Q-values.
        alphasau_agent = DQNAgent(0, config=config, training=False)
        alphasau_agent.load_weights(as_checkpoint_path)
        p0 = Player("0-AlphaSau", agent=alphasau_agent)
    elif agent_choice == "user":
        p0 = Player("0-User", agent=GUIAgent(0))
    elif agent_choice == "rule":
        p0 = Player("0-Hans", agent=RuleBasedAgent(0))
    elif agent_choice == "static":
        p0 = Player("0-Static", agent=StaticPolicyAgent(0))
    else:
        p0 = Player("0-RandomGuy", agent=RandomCardAgent(0))

    # Players 1-3 are RuleBasedAgents.
    players = [
        p0,
        Player("1-Zenzi", agent=RuleBasedAgent(1)),
        Player("2-Franz", agent=RuleBasedAgent(2)),
        Player("3-Andal", agent=RuleBasedAgent(3))
    ]

    # Rig the game so Player 0 has the cards to play a Herz-Solo.
    # Also, force them to play it.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    # The GUI initializes PyGame and registers on events provided by the controller. Everything single-threaded.
    #
    # The controller runs the game as usual. Whenever the GUI receives an event, it can block execution, so the controller must wait
    # for the GUI to return control. Until then, it can draw stuff and wait for user input (mouse clicks, card choices, ...).
    logger.info("Starting GUI.")
    with Gui(controller.game_state) as gui:
        # Run an endless loop of single games.
        logger.info("Starting game loop...")
        try:
            while True:
                controller.run_game()
        except UserQuitGameException:           # Closing the window or pressing [Esc]
            logger.info("User quit game.")

    logger.info("Shutdown.")


if __name__ == '__main__':
    main()
