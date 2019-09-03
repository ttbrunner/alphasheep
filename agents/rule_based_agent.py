from typing import List, Iterable
import numpy as np

from agents.agents import PlayerAgent
from game.card import Card, Suit, Pip, pip_scores
from game.game_mode import GameMode, GameContract
from log_util import get_class_logger


class RuleBasedAgent(PlayerAgent):
    """
    Agent that plays according to a number of fixed "rules" that mirror most of the author's knowledge of the game :)
    Naturally, there are lots of special constellations that are not accounted for here, but this agent should be able to play
    like a beginner-level human.

    Right now, it's a loose assortment of heuristics.
    Supports only suit-solo with the agent as the declaring player.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)

        self.logger = get_class_logger(self)

        # "Power" values for quickly determining which card can beat which.
        # Defining this here because we don't want to be dependent on the enum int values.
        self._suit_power = {Suit.eichel: 40, Suit.gras: 30, Suit.herz: 20, Suit.schellen: 10}
        self._pip_power = {Pip.sau: 8, Pip.zehn: 7, Pip.koenig: 6, Pip.ober: 5, Pip.unter: 4, Pip.neun: 3, Pip.acht: 2, Pip.sieben: 1}

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode) -> Card:

        if game_mode.contract != GameContract.suit_solo:
            raise NotImplementedError("Sorry, can only play a Suit-solo right now.")

        # Are we the main player?
        declaring = game_mode.declaring_player_id == self.player_id
        if not declaring:
            raise NotImplementedError("Sorry, can only play a Suit-solo as the declaring player right now.")

        # We have a set of high-level actions, such as "play highest trump", "play low color" etc.
        # Right now, this is only for logging purposes, but ultimately we would want some sort of hierarchical planning:
        # First, choose an action, and later find a card that fits.
        action = None

        valid_cards = [c for c in cards_in_hand if game_mode.is_play_allowed(c, cards_in_hand, cards_in_trick)]
        own_trumps = self._trumps_by_power(in_cards=valid_cards, game_mode=game_mode)

        if len(cards_in_trick) == 0:
            # We are leading.
            if any(own_trumps):
                # As a general rule, we'd like to play our trumps high to low.
                action = "play_highest_trump"
                selected_card = own_trumps[-1]
            else:
                # No trump, play color.
                saus = [c for c in cards_in_hand if c.pip == Pip.sau]
                if any(saus):
                    # Play a color sau.
                    action = "play_color_sau"
                    selected_card = saus[np.random.randint(len(saus))]
                else:
                    # Play a Spatz (low value).
                    # Depending on what happend in the game, it might be very important which color is played.
                    # But this goes beyond this simple baseline :)
                    action = "play_spatz"
                    selected_card = self._cards_by_value(valid_cards)[0]

        else:
            # Not leading.
            # Do we need to match? Get all valid options.
            c_lead = cards_in_trick[0]

            if any(c for c in valid_cards if game_mode.is_trump(c)):
                # We can play trump (in fact, any trump we have).  Which one will we pick?

                # Find out if we can beat the preceding cards.
                if any(game_mode.is_trump(c) for c in cards_in_trick):
                    beating_cards = [c for c in own_trumps
                                     if self._trump_power(c) > max(self._trump_power(c2) for c2 in cards_in_trick if game_mode.is_trump(c2))]
                else:
                    beating_cards = own_trumps

                if any(beating_cards):
                    # We can beat the preceding cards.
                    # a) pick lowest trump that will beat the prev players. Do this if 0-1 players come after us.
                    # b) pick a high trump to prevent following players to beat. Do this if 2 players come after us.
                    beat_low = len(cards_in_trick) > 1
                    beating_cards_by_power = self._trumps_by_power(beating_cards, game_mode)
                    if beat_low:
                        action = "beat_trump_low"
                        selected_card = beating_cards_by_power[0]
                    else:
                        action = "beat_trump_high"
                        selected_card = beating_cards_by_power[-1]
                else:
                    # We actually cannot beat them. Play a spatz (low-value card).
                    action = "play_spatz"
                    selected_card = self._cards_by_value(valid_cards)[0]

            elif not game_mode.is_trump(c_lead):
                # We can't play trump, but the leading card also is not a trump.
                # Therefore we might beat it with a higher card of the same suit.
                beating_cards = [c for c in valid_cards if c.suit == c_lead.suit
                                 and self._pip_power[c.pip] > max(self._pip_power[c2.pip] for c2 in cards_in_trick if c2.suit == c_lead.suit)]
                if any(beating_cards):
                    # In the case of suit, always beat high (hopefully with a sau).
                    action = "beat_suit_high"
                    selected_card = sorted(beating_cards, key=lambda c: self._pip_power[c.pip], reverse=True)[0]
                else:
                    action = "play_spatz"
                    selected_card = self._cards_by_value(valid_cards)[0]

            else:
                # We cannot beat it with or without trump. Play a spatz (low-value card).
                action = "play_spatz"
                selected_card = self._cards_by_value(valid_cards)[0]

        self.logger.debug(f'Executing action "{action}".')
        assert game_mode.is_play_allowed(selected_card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick)
        return selected_card

    # ========
    # Helper functions for quick comparison of trumps and cards.
    # We might want to move them to GameMode as right now they are partially dependent on this being a suit solo.
    # ========

    def _trump_power(self, c: Card):
        val = 0
        if c.pip == Pip.ober or c.pip == Pip.unter:
            if c.pip == Pip.ober:  # Ober is always higher than unter
                val += 100
            val += 100 + self._suit_power[c.suit]       # Ober or unter are ranked by suit
        else:
            val = self._pip_power[c.pip]                # Plain trump color is ranked by pip
        return val

    def _trumps_by_power(self, in_cards: Iterable[Card], game_mode: GameMode) -> List[Card]:
        # Filters in_cards by trumps and returns them, sorted py power.
        return sorted([c for c in in_cards if game_mode.is_trump(c)], key=self._trump_power)

    def _cards_by_value(self, in_cards: Iterable[Card]) -> List[Card]:
        # Sorts cards by value.
        return sorted(in_cards, key=lambda c: pip_scores[c.pip])
