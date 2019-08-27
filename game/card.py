"""
WARN: DO NOT CHANGE THE ENUMS IN THIS FILE!
Changing the values might affect the order of the state/action space of agents, and will break compatibility with previously
saved model checkpoints.
"""

from enum import IntEnum


class Suit(IntEnum):
    schellen = 0
    herz = 1
    gras = 2
    eichel = 3

    def __str__(self):
        return self.name


class Pip(IntEnum):
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

        # There are some performance problems doing enum lookups, apparently Python implements them in a bit of a convoluted way.
        # We often use Card as a dict key, so this has turned out to be a bit problematic. It turns out to be much faster
        # to just precalc a unique card ID instead of comparing suits and pips (Python 3.5).
        self._unique_hash = hash(Card) * 23 + self.suit.value * 23 + self.pip.value

    def __str__(self):
        return "({} {})".format(self.suit.name, self.pip.name)

    def __eq__(self, other):
        return isinstance(other, Card) and self._unique_hash == other._unique_hash

    def __hash__(self):
        return self._unique_hash


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
    """ Returns an ordered deck. """
    return [Card(suit, pip) for suit in Suit for pip in Pip]
