"""
Runs a single game with an interactive GUI. Use for debugging purposes, or just to see the agents play.
"""
import argparse

from controller.game_controller import GameController
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

    # GUI integration: GUI starts the controller, takes a reference to the GameState and that's that.
    # The GUI registers on events provided by the controller. Since everything runs single-threaded,
    # it can block (and wait for user clicks) before the controller continues with the next move.
    #
    # In this way, the GUI can be used to debug and watch a single game.

    controller = GameController(players)
    if run_with_gui:
        gui = Gui(controller.game_state)
        gui.start()

    # Run a single game before terminating.
    controller.run_game()


if __name__ == '__main__':
    main()
