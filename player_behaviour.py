from abc import ABC, abstractmethod
from typing import Iterable, List
import numpy as np

from card import Card


class PlayerBehavior(ABC):
    """
    Abstract class for all types of agents.
    """

    @abstractmethod
    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card]):
        """
        Returns the card which the agent wants to play.
        :param cards_in_hand: the cards which the player currently has in hand.
        :param cards_in_trick: cards that previous players have put into the trick (if any). If empty, then the player is leading.
        :return: A card that is contained in cards_in_hand.
        """
        pass


class RandomCardAgent(PlayerBehavior):
    # This agent selects a random card from its hand and plays it (ignoring any rules).

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card]):
        cards_in_hand = list(cards_in_hand)
        return cards_in_hand[np.random.randint(len(cards_in_hand))]
