"""
Runs a single game with an interactive GUI. Use for debugging purposes, or just to see the agents play.
"""
import argparse

from controller.dealing_behavior import DealWinnableHand
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameMode, GameContract
from game.game_state import Player

from gui.gui import Gui
from agents.agents import RandomCardAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_gui", help="Run without the GUI (not interactively).", action="store_true")
    args = parser.parse_args()
    run_with_gui = not args.disable_gui

    players = [
        Player("0-Hans", agent=RandomCardAgent()),
        Player("1-Zenzi", agent=RandomCardAgent()),
        Player("2-Franz", agent=RandomCardAgent()),
        Player("3-Andal", agent=RandomCardAgent())
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
    if run_with_gui:
        gui = Gui(controller.game_state)

    # Run a single game before terminating.
    controller.run_game()


if __name__ == '__main__':
    main()
