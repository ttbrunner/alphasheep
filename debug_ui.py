import numpy as np

from card import new_deck, Suit
from game import GameMode, GameType, GameState
from gui.gui import Gui

from player import Player


def main():

    # Create game state
    players = [Player(0, "Hans"), Player(1, "Zenzi"), Player(2, "Franz"), Player(3, "Andal")]
    game_state = GameState(players)

    # No game selection for now: always plaing Herz-Solo.
    game_state.game_mode = GameMode(GameType.solo, trump_suit=Suit.herz)

    # Deal random cards
    deck = new_deck()
    np.random.shuffle(deck)
    i_deck = 0
    for p in players:
        p.cards_in_hand.extend(deck[i_deck:i_deck + 8])
        i_deck += 8

    # Show the UI
    gui = Gui(game_state=game_state)
    gui.run()


if __name__ == '__main__':
    main()
