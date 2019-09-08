from enum import Enum
from typing import Iterable, List

from simulator.card_defs import Card, Pip, Suit


class GameContract(Enum):
    rufspiel = 0,
    wenz = 1,
    suit_solo = 2,
    # No tout/sie for now

    def __str__(self):
        return self.name


class GameMode:
    """
    The Game Mode stores all info about the variant of the game that is being played (contract, trump suit, ruf suit, ...)
    and provides logic to enforce the game rules that apply.
    """

    def __init__(self, contract: GameContract, declaring_player_id, ruf_suit: Suit = None, trump_suit: Suit = None):
        # Here are a couple of checks that are just for data integrity.
        if contract == GameContract.rufspiel:
            assert ruf_suit is not None
            assert trump_suit is None or trump_suit == Suit.herz, "Invalid trump suit for Rufspiel: {}".format(trump_suit)
            trump_suit = Suit.herz
        else:
            assert ruf_suit is None

        if contract == GameContract.suit_solo:
            assert trump_suit is not None

        if contract == GameContract.wenz:
            assert trump_suit is None, "No Farbwenz allowed, you Breznsalzer!"

        self.contract = contract
        self.declaring_player_id = declaring_player_id          # Only storing ID, so agents can't directly access other Player objects.
        self.trump_suit = trump_suit
        self.ruf_suit = ruf_suit

    def __str__(self):
        if self.contract == GameContract.suit_solo:
            return "({} solo)".format(self.trump_suit.name)
        elif self.contract == GameContract.rufspiel:
            return "(rufspiel: auf die {} sau)".format(self.ruf_suit.name)
        else:
            return "({})".format(self.contract)

    def is_trump(self, card: Card) -> bool:
        """
        Returns true if a card is trump in this game variant.
        """
        return card.suit == self.trump_suit \
            or (card.pip == Pip.ober and self.contract != GameContract.wenz) \
            or card.pip == Pip.unter

    def is_play_allowed(self, card: Card, cards_in_hand: Iterable[Card], cards_in_trick: List[Card]) -> bool:
        """
        Returns true if a player is allowed to play a specific card.
        :param card: the card to be played.
        :param cards_in_hand: all cards (including the card) in the Player's hand.
        :param cards_in_trick: all cards in the current trick (excluding the card). Can be empty.
        :return: True if the card can be played under the game rules.
        """

        assert card in cards_in_hand
        cards_in_hand = list(cards_in_hand)

        rufsau = None
        if self.contract == GameContract.rufspiel:
            rufsau = Card(suit=self.ruf_suit, pip=Pip.sau)

        # To make matching easier, we redefine suits as follows:
        # - All trumps are assigned to a special "trump suit", and this includes unter and ober (depending on the variant).
        # - Trump cards do not belong to their original suits (e.g. the suit of "Gras Unter" is not Gras, but Trump).
        # - Since all trumps need to be matched with other trumps, we can simply match everything by suit (no matter if trump or not).
        def true_suit(c: Card) -> int:
            return 9001 if self.is_trump(c) else c.suit.value

        if len(cards_in_trick) == 0:
            # Player is leading.
            if self.contract == GameContract.rufspiel and card.suit == self.ruf_suit:
                # Player is playing ruf-suit.
                if rufsau in cards_in_hand:
                    # Player has the Rufsau. In that case, they are not allowed to play any card of ruf-suit unless:
                    if len(cards_in_hand) == 1:
                        # If it's the only card left. TODO: or was it 2 instead of 1?
                        return True
                    elif sum(c for c in cards_in_hand if true_suit(c) == true_suit(card)) >= 4:
                        # They have 4 cards of the ruf-suit, which enables the "davonlaufen" maneuver.
                        # TODO: Check exact rules of Davonlaufen again. This leads to much argument in real life as well :)
                        return True
                    else:
                        # Otherwise, not allowed to play the ruf-suit.
                        return False
            # Leading with any card of any other suit is OK.
            return True

        # Player is not leading, so they have to match the first card.
        first_card = cards_in_trick[0]

        if true_suit(card) == true_suit(first_card):
            # Player is matching suit.
            if self.contract == GameContract.rufspiel and card.suit == self.ruf_suit:
                # Player is matching the ruf-suit. If they have the ruf-sau, then they need to play it.
                if card != rufsau and rufsau in cards_in_hand:
                    # Player has the ruf-sau but did not play it!
                    # TODO: Check exact rules of Davonlaufen again! Can we do Davonlaufen while matching?
                    return False
            # All other cases of matching are OK.
            return True

        if any(c for c in cards_in_hand if true_suit(c) == true_suit(first_card)):
            # Player is not matching suit when they could.
            return False

        # Finally - player is not leading and not matching because they can't.
        if card == rufsau and len(cards_in_hand) > 1:
            # One last rule - not allowed to "schmier" the Rufsau if there is any other choice.
            return False

        return True

    def get_trick_winner(self, cards_in_trick: List[Card]) -> int:
        """
        Determines the index of the winning card in a trick.
        :param cards_in_trick: cards in a complete trick (must be of length 4).
        :return: the index (into the list) of the winning card.
        """

        assert len(cards_in_trick) == 4

        # Assign values to all cards, and then pick the highest.
        # Ober:             1210 - 1240
        # Unter:            1110 - 1140
        # Trump suit:       1001 - 1008
        # Non-trump suit:      1 -    8 (only if same suit as first card)

        # Repeating these here because scoring should not depend on enum definitions.
        suit_vals = {Suit.eichel: 40, Suit.gras: 30, Suit.herz: 20, Suit.schellen: 10}
        pip_vals = {Pip.sau: 8, Pip.zehn: 7, Pip.koenig: 6, Pip.ober: 5, Pip.unter: 4, Pip.neun: 3, Pip.acht: 2, Pip.sieben: 1}

        is_trump_first = self.is_trump(cards_in_trick[0])

        i_highest = -1
        val_highest = -1
        for i_c, c in enumerate(cards_in_trick):
            value = 0
            if self.is_trump(c):
                # Bonus points for being trump
                value += 1000
                if c.pip == Pip.unter:
                    value += 100 + suit_vals[c.suit]                    # More bonus for being trump
                elif c.pip == Pip.ober:
                    value += 200 + suit_vals[c.suit]                    # Even more bonus for being trump, making Schafkopf great again (sorry)
                else:
                    value += pip_vals[c.pip]                            # Trump-suit cards are ranked according to pip.
            else:
                # Not trump = not so great :(
                if not is_trump_first and c.suit == cards_in_trick[0].suit:
                    # If the first card is also a non-trump suit (and this one matches it), then the higher pip wins.
                    value += pip_vals[c.pip]

            if value > val_highest:
                val_highest = value
                i_highest = i_c

        assert i_highest >= 0
        return i_highest
