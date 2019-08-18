"""
Runs a large number of games without the GUI. Use this to train an agent.
"""
from agents.agents import RandomCardAgent
from controller.game_controller import GameController
from game.game_state import Player


def main():
    # For starters, run a single game with a RL mock agent.
    players = [
        Player("0-Hans", agent=RandomCardAgent()),
        Player("1-Zenzi", agent=RandomCardAgent()),
        Player("2-Franz", agent=RandomCardAgent()),
        Player("3-Andal", agent=RandomCardAgent())
    ]
    controller = GameController(players)
    controller.run_game()


if __name__ == '__main__':
    main()
