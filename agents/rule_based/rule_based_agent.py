from typing import List, Iterable
import numpy as np

from agents.agents import PlayerAgent
from game.card import Card, Suit, Pip, pip_scores
from game.game_mode import GameMode, GameContract
from utils.log_util import get_class_logger


class RuleBasedAgent(PlayerAgent):
    """
    Agent that plays according to a number of fixed "rules" that mirror most of the author's knowledge of the game :)
    Naturally, there are lots of special constellations that are not accounted for here, but this agent should be able to play
    like a beginner-level human.

    Right now, it's a loose assortment of heuristics, and LOTS of if-else - many of them redundant.
    CC is probably over 9000, sorry for creating an abomination. Maybe we should call it IfElseAgent :)

    Supports only suit-solo right now.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)

        self.logger = get_class_logger(self)

        # "Power" values for quickly determining which card can beat which.
        # Defining this here because we don't want to be dependent on the enum int values.
        self._suit_power = {Suit.eichel: 40, Suit.gras: 30, Suit.herz: 20, Suit.schellen: 10}
        self._pip_power = {Pip.sau: 8, Pip.zehn: 7, Pip.koenig: 6, Pip.ober: 5, Pip.unter: 4, Pip.neun: 3, Pip.acht: 2, Pip.sieben: 1}

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode) -> Card:
        # For now, this function is a dispatcher that invokes individual behaviors based on the game mode.
        # The (almost hardcoded) behavior in these functions is highly redundant, but keeping it this way
        # hopefully makes it more readable, debuggable, and understandable.
        # If we develop any ambitions about making this agent play REALLY well, we might have to consolidate this.

        if game_mode.contract == GameContract.suit_solo:
            # Are we the main player?
            if game_mode.declaring_player_id == self.player_id:
                selected_card = self._play_card_solo_declaring(cards_in_hand, cards_in_trick, game_mode)
            else:
                selected_card = self._play_card_solo_not_declaring(cards_in_hand, cards_in_trick, game_mode)
        else:
            raise NotImplementedError("Sorry, can only play a Suit-solo right now.")

        return selected_card

    def _play_card_solo_declaring(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode) -> Card:
        # When a solo is being played and we are the declaring player.

        # We have a set of high-level actions, such as "play highest trump", "play low color" etc.
        # Right now, this is only for logging purposes, but ultimately we would want some sort of hierarchical planning:
        # First, choose an action, and later find a card that fits.
        # These action definitions could also be shared across behaviors, so this could remove some of the redundancy
        #  we get when duplicating behavior for different game modes.

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

    def _play_card_solo_not_declaring(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode) -> Card:
        # When a solo is being played and the declaring player is the enemy.
        valid_cards = [c for c in cards_in_hand if game_mode.is_play_allowed(c, cards_in_hand, cards_in_trick)]
        own_trumps = self._trumps_by_power(in_cards=valid_cards, game_mode=game_mode)
        non_trumps = set(valid_cards).difference(own_trumps)

        if len(cards_in_trick) == 0:
            # We are leading.
            # As a general rule, we want to play any non-trump Sau we have (IF this suit has not been played before).
            # We hope that the enemy has a card of this suit and is forced to match.
            # TODO: create memory, and check if a) the suit has been played before, b) the enemy is known to have it
            # TODO: in this case, do NOT play it. There are exceptions, but well.
            saus = [c for c in non_trumps if c.pip == Pip.sau]
            if any(saus):
                action = "play_color_sau"
                selected_card = saus[np.random.randint(len(saus))]
            else:
                # No sau: don't play 10 etc., rather play a small card and hope our partners have the sau
                action = "play_spatz"
                selected_card = self._cards_by_value(non_trumps if any(non_trumps) else own_trumps)[0]

        else:
            # Not leading.
            # In this situation, we want to:
            # - minimize damage as the enemy will probably take the trick
            # - maximize score whenever it looks like we (or our partners) might take it.

            # Has the enemy already played their card?
            # TODO: This is HORRIBLE and I'm tired. Make a nice helper function and abstract this modulo crap away for all eternity, PLEASE!
            enemy_id = game_mode.declaring_player_id
            enemy_card_id = None
            i_p = self.player_id
            for i in range(len(cards_in_trick)):
                i_p = (i_p - 1) % 4
                if i_p == enemy_id:
                    enemy_card_id = len(cards_in_trick) - 1 - i
                    break
            enemy_card = cards_in_trick[enemy_card_id] if enemy_card_id is not None else None

            if enemy_card_id is not None:
                # The enemy has already made their move.
                if self._winning_card(cards_in_trick, game_mode) != enemy_card:
                    # A partner has already beaten the enemy.
                    # Give them as many points as possible.
                    non_trump_by_value = self._cards_by_value(non_trumps)
                    if any(non_trump_by_value):
                        # Put the most expensive non-trump
                        action = "schmier_points"
                        selected_card = non_trump_by_value[-1]
                    else:
                        # We are not allowed to schmier a non-trump. Put the most expensive trump.
                        # TODO: don't schmier an ober!
                        action = "schmier_trump"
                        selected_card = self._cards_by_value(own_trumps)[-1]

                else:
                    # The enemy has already played, but it's not clear who will take the trick.
                    # Can we beat it?
                    beating_cards = [c for c in valid_cards if c == self._winning_card(cards_in_trick + [c], game_mode)]
                    if any(beating_cards):
                        # We can actually beat the enemy. Use the most expensive option.
                        # TODO: don't schmier-stech with ober if not necessary!
                        action = "beat_expensive"
                        selected_card = self._cards_by_value(beating_cards)[-1]
                    else:
                        # Can't beat them. Can we expect a partner to beat them?
                        # TODO: create memory of cards that are still in the game. Recognize if the enemy played the currently highest trump.
                        # Right now, we don't think a partner could ever beat the enemy. With that memory, if the enemy played a low trump, we could
                        # wager on our partners in some cases.
                        # Play a Spatz.
                        action = "play_spatz"
                        selected_card = self._cards_by_value(valid_cards)[0]
            else:
                # The enemy has not yet played.
                if game_mode.is_trump(cards_in_trick[0]):
                    # WTF, our partner played trump. Idiot! We expect the enemy will surely take it.
                    action = "play_spatz;insult_leader"
                    selected_card = self._cards_by_value(valid_cards)[0]
                else:
                    # It's a suit card. As a general rule, we hope that the enemy has to match and we might take it.
                    # This might not be the case depending on what suits were already played, but we are not that smart right now :)
                    beating_cards = [c for c in valid_cards if c == self._winning_card(cards_in_trick + [c], game_mode)]
                    if any(beating_cards):
                        # We can beat our partners.
                        if game_mode.is_trump(beating_cards[0]):
                            # We don't have that suit and can beat it with a trump.
                            # There is a lot of options here, but we will stick to the golden rule: "Mi'm Unter gehst net unter".
                            beating_unter = [c for c in valid_cards if c.pip == Pip.unter]
                            if any(beating_unter):
                                # Play the lowest unter, which makes the trick "safe" (the enemy can't beat with an expensive sau).
                                action = "beat_with_unter"
                                selected_card = self._trumps_by_power(beating_unter, game_mode)[0]
                            else:
                                # It's also fine to be risky and beat with sau/zehn. We expect the enemy to take those away soon anyway.
                                action = "beat_expensive"
                                selected_card = self._cards_by_value(beating_cards)[-1]
                        else:
                            # We must match the suit and can beat our partner.
                            # Do it, since this probably means playing the sau, which is good.
                            action = "match_expensive"
                            selected_card = self._cards_by_value(valid_cards)[-1]
                    else:
                        # The partner lead is non-trump and we can't beat the partners.
                        # TODO: special situation: did the 2nd partner beat with a trump,
                        #  so high that the enemy won't be able to take it? then schmier.
                        if any(c for c in cards_in_trick if c.suit == cards_in_trick[0] and c.pip == Pip.sau):
                            # One of the partners played the suit-sau. We hope the enemy needs to match!
                            # TODO: don't schmier if it's clear from memory that the enemy can beat it.
                            action = "schmier_points"
                            selected_card = self._cards_by_value(valid_cards)[-1]
                        else:
                            # It's non-trump, low, we need to match, looks bad man.
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
            if c.pip == Pip.ober:                       # Ober is always higher than unter
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

    def _winning_card(self, cards_in_trick: List[Card], game_mode: GameMode) -> Card:
        # Gets the winning card out of a trick. The trick can have less than 4 cards.
        assert any(cards_in_trick)

        trumps = self._trumps_by_power(cards_in_trick, game_mode)
        if any(trumps):
            # The highest trump wins.
            return trumps[-1]
        else:
            # The highest card of the suit of the first card wins.
            suit = cards_in_trick[0].suit
            suit_cards = sorted((c for c in cards_in_trick if c.suit == suit), key=lambda c: self._pip_power[c.pip])
            return suit_cards[-1]
