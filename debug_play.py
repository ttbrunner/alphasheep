"""
Terms

Round: set of games
Game: all tricks are played for one game to conclude
Trick: individual "Stich" (one card from each player)
"""

from controller.game_controller import GameController
from game import Player
import numpy as np

from gui.gui import Gui
from player_behaviour import RandomCardAgent


def main():
    players = [
        Player("Hans", behavior=RandomCardAgent()),
        Player("Zenzi", behavior=RandomCardAgent()),
        Player("Franz", behavior=RandomCardAgent()),
        Player("Andal", behavior=RandomCardAgent())
    ]

    # Idea for GUI integration: GUI starts the controller, takes a reference to the GameState and that's that.
    # 2 options:
    # a) Either the controller notifies observers (GUI) at specific points (e.g. after a card has been played)
    # b) Or we run in a multi-threaded fashion and the GUI constantly monitors the GameState
    #
    # I'd like to do a) in an Observer pattern, since then we can run single-threaded without much trouble.
    # This setup implies that the GUI blocks and can delay the game at any time. For our purpose (dev debugs a game, or user plays
    # against the AI), this is actually preferable.
    #
    # b) might be better if we want to observe in real time, or if we want to do longer animations in the GUI while the game is running
    # in the background. For simplicity, let's do a).

    controller = GameController(players)
    gui = Gui(controller.game_state)
    gui.start()

    controller.run_game()


if __name__ == '__main__':
    main()
