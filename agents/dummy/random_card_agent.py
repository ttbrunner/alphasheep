from typing import Iterable, List

import numpy as np

from simulator.player_agent import PlayerAgent
from simulator.card_defs import Card
from simulator.game_mode import GameMode


class RandomCardAgent(PlayerAgent):
    """
    Dummy agent. Selects a random card from its hand and plays it.
    """

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):
        # Shuffles the agent's cards and picks the first one that is allowed.

        cards_in_hand = list(cards_in_hand)
        np.random.shuffle(cards_in_hand)
        for card in cards_in_hand:
            if game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                return card

        raise ValueError("None of the Player's cards seem to be allowed! This should never happen! Player has cards: {}".format(
            ",".join(str(c) for c in cards_in_hand)))
