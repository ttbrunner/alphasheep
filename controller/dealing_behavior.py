from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np

from game.card import new_deck, Card, Suit, Pip
from game.game_mode import GameMode, GameContract


class DealingBehavior(ABC):
    """
    We can exchange dealing behavior for easier experiments: if we want the agent to play specific game variants exclusively.
    """

    @abstractmethod
    def deal_hands(self) -> List[Iterable[Card]]:
        """
        Deals 4 hands (one for each player) of 8 cards each.
        :return: list(4) of iterable(8).
        """
        pass


class DealFairly(DealingBehavior):
    """
    Default / baseline dealer - randomly shuffles the deck and deals the cards.
    """

    def deal_hands(self) -> List[Iterable[Card]]:
        deck = new_deck()
        np.random.shuffle(deck)

        player_hands = [set(deck[i*8:(i+1)*8]) for i in range(4)]
        return player_hands


class DealWinnableHand(DealingBehavior):
    """
    This dealer is cheating - they always make sure that player X can play a specific game!
    """

    def __init__(self, game_mode: GameMode):
        assert game_mode.declaring_player_id is not None
        self._game_mode = game_mode

    def deal_hands(self) -> List[Iterable[Card]]:
        deck = new_deck()

        # Repeat random shuffles until the player's cards are good enough.
        while True:
            np.random.shuffle(deck)
            player_hands = [set(deck[i * 8:(i + 1) * 8]) for i in range(4)]
            if self._are_cards_suitable(player_hands[self._game_mode.declaring_player_id], self._game_mode):
                return player_hands

    def _are_cards_suitable(self, cards_in_hand, game_mode: GameMode):
        # Quick and dirty heuristic for deciding whether to play a solo.

        if game_mode.contract != GameContract.suit_solo:
            raise NotImplementedError("Only Suit-solo is allowed at this time.")

        # Needs 6 trumps and either good Obers or lots of Unters.
        if sum(1 for c in cards_in_hand if game_mode.is_trump(c)) >= 6:
            if sum(1 for c in cards_in_hand if c.pip == Pip.ober) >= 2:
                return True
            elif sum(1 for c in cards_in_hand if c.pip == Pip.unter) >= 3:
                return True
            elif Card(Suit.eichel, Pip.ober) in cards_in_hand:
                return True
        return False


class DealExactly(DealingBehavior):
    """
    Deals exactly the specified cards.
    """
    def __init__(self, player_hands: List[Iterable[Card]]):
        assert len(player_hands) == 4 and not any(cards for cards in player_hands if len(list(cards)) != 8)
        self.player_hands = player_hands

    def deal_hands(self) -> List[Iterable[Card]]:
        # Create new list/sets to prevent modification
        return [set(cards) for cards in self.player_hands]
