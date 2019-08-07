"""
Terms

Round: set of games
Game: all tricks are played for one game to conclude
Trick: individual "Stich" (one card from each player)
"""

from card import new_deck
from player import Player
import numpy as np


def main():
    np.random.seed(0)
    players = [Player(0, "Hans"), Player(1, "Zenzi"), Player(2, "Franz"), Player(3, "Andal")]

    print("New game.")

    # Reset players (take away their cards from a previous game)
    for p in players:
        assert len(p.cards_in_hand) == 0
        p.cards_in_scored_tricks.clear()

    print("Dealing.")

    # New deck, shuffle.
    # No cutting necessary in this simulation since the deck is already perfectly shuffled
    deck = new_deck()
    np.random.shuffle(deck)

    # Deal to Players. First 4 cards
    i_deck = 0
    for p in players:
        p.cards_in_hand.extend(deck[i_deck:i_deck + 4])
        i_deck += 4

    # Second 4 cards. Before this, "laying" is possible (not implemented right now)
    for p in players:
        p.cards_in_hand.extend(deck[i_deck:i_deck + 4])
        i_deck += 4

    for p in players:
        print("Player {} has cards:".format(p))
        print("\n".join("\t{}".format(c) for c in p.cards_in_hand))


if __name__ == '__main__':
    main()
