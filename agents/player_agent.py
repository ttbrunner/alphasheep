from abc import ABC, abstractmethod
from typing import Iterable, List
import numpy as np

from game.card import Card
from game.game_mode import GameMode


class PlayerAgent(ABC):
    """
    Abstract class for all types of agents.
    """

    @abstractmethod
    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):
        """
        Returns the card which the agent wants to play.
        :param cards_in_hand: the cards which the player currently has in hand.
        :param cards_in_trick: cards that previous players have put into the trick (if any). If empty, then the player is leading.
        :param game_mode: the game mode that is currently being played.
        :return: A card that is contained in cards_in_hand.
        """
        pass


class RandomCardAgent(PlayerAgent):
    # Baseline agent. Selects a random card from its hand and plays it.

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):
        # Shuffles the agent's cards and picks the first one that is allowed.

        cards_in_hand = np.array(cards_in_hand)
        np.random.shuffle(cards_in_hand)
        for card in cards_in_hand:
            if game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                return card

        raise ValueError("None of the Player's cards seem to be allowed! This should never happen! Player has cards: {}".format(
            ",".join(str(c) for c in cards_in_hand)))