import numpy as np

from card import new_deck
from gui.gui import Gui

from player import Player


def main():
    # Create players
    players = [Player(0, "Hans"), Player(1, "Zenzi"), Player(2, "Franz"), Player(3, "Andal")]

    # Deal random cards
    deck = new_deck()
    np.random.shuffle(deck)
    i_deck = 0
    for p in players:
        p.cards_in_hand.extend(deck[i_deck:i_deck + 8])
        i_deck += 8

    # Show the UI
    gui = Gui(players=players)
    gui.run()


if __name__ == '__main__':
    main()
