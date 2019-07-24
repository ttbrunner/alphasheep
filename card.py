from enum import Enum
import numpy as np

class Suit(Enum):
    schellen = 0
    herz = 1
    gras = 2
    eichel = 3

    def __str__(self):
        return self.name


class Pip(Enum):
    sieben = 1
    acht = 2
    neun = 3
    unter = 4
    ober = 5
    koenig = 6
    zehn = 7
    sau = 8

    def __str__(self):
        return self.name


class Card:
    def __init__(self, suit: Suit, pip: Pip):
        self.suit = suit
        self.pip = pip

    def __str__(self):
        return "({} {})".format(self.suit, self.pip)

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.pip == other.pip

    def __hash__(self):
        return hash(self.suit) * 23 + hash(self.pip.value)


pip_scores = {
    Pip.sieben: 0,
    Pip.acht: 0,
    Pip.neun: 0,
    Pip.unter: 2,
    Pip.ober: 3,
    Pip.koenig: 4,
    Pip.zehn: 10,
    Pip.sau: 11}


def new_deck():
    # Returns an ordered deck.
    return [Card(suit, pip) for suit in Suit for pip in Pip]


# DEBUG
if __name__ == '__main__':
    score_all = sum(pip_scores[c.pip] for c in new_deck())
    print(score_all)
