from typing import Iterable, List

from agents.agents import PlayerAgent
from game.card import Card, Suit, Pip, new_deck
from game.game_mode import GameMode


class StaticPolicyAgent(PlayerAgent):
    """
    Dummy agent. Has a list of cards, ranked by preference, and plays them whenever it can.
    I had this idea when observing that DQNAgent would sometimes learn a completely static policy,
    notably one that significantly outperforms RandomCardAgent.

    This policy is extracted from one of those agents (more precisely: I copypasted the Q-vector from the log output).
    Surprisingly, it performs pretty well in a solo situation (with alot of trump)!
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)

        self.static_policy = [
            Card(Suit.eichel, Pip.ober),
            Card(Suit.eichel, Pip.unter),
            Card(Suit.gras, Pip.neun),
            Card(Suit.gras, Pip.ober),
            Card(Suit.herz, Pip.neun),
            Card(Suit.eichel, Pip.acht),
            Card(Suit.herz, Pip.acht),
            Card(Suit.herz, Pip.koenig),
            Card(Suit.gras, Pip.sau),
            Card(Suit.gras, Pip.unter),
            Card(Suit.schellen, Pip.ober),
            Card(Suit.herz, Pip.unter),
            Card(Suit.schellen, Pip.unter),
            Card(Suit.eichel, Pip.sau),
            Card(Suit.schellen, Pip.sau),
            Card(Suit.schellen, Pip.koenig),
            Card(Suit.herz, Pip.ober),
            Card(Suit.herz, Pip.sieben),
            Card(Suit.gras, Pip.zehn),
            Card(Suit.eichel, Pip.neun),
            Card(Suit.schellen, Pip.zehn),
            Card(Suit.gras, Pip.sieben),
            Card(Suit.herz, Pip.sau),
            Card(Suit.schellen, Pip.sieben),
            Card(Suit.eichel, Pip.sieben),
            Card(Suit.schellen, Pip.neun),
            Card(Suit.herz, Pip.zehn),
            Card(Suit.eichel, Pip.koenig),
            Card(Suit.schellen, Pip.acht),
            Card(Suit.gras, Pip.koenig),
            Card(Suit.eichel, Pip.zehn),
            Card(Suit.gras, Pip.acht),
        ]
        assert len(set(self.static_policy)) == len(new_deck()), "Need to include all cards!"

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):
        # Plays the first card from the static policy that is allowed.

        for card in self.static_policy:
            if card in cards_in_hand and game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                return card

        raise ValueError("None of the Player's cards seem to be allowed! This should never happen! Player has cards: {}".format(
            ",".join(str(c) for c in cards_in_hand)))
