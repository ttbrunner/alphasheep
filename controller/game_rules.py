from typing import Iterable, List

from card import Card, Pip
from game import GameVariant


class GameRules:
    # A rule provider that can be used both by the controller to query the following things:
    # - Whether an action (i.e. playing a specific card) is allowed in this situation
    # - Who wins a trick
    #
    # At first, the AI players can also use this class to limit their actions to only valid moves.
    # Later, we could allow RL agents to play freely and be punished for invalid moves.
    #
    # In any case, after every action by a player, the controller should double-check whether the action was allowed.

    def __init__(self, game_variant: GameVariant):
        self.game_variant = game_variant

    def is_card_allowed(self, card: Card, cards_in_hand: Iterable[Card], cards_in_trick: List[Card]):
        # Returns true if a player is allowed to play a specific card.

        if card not in cards_in_hand:
            return False

        # To make matching easier, we redefine suits as follows:
        # - All trumps are assigned to a special "trump suit", and this includes unter and ober (depending on the variant).
        # - Trump cards do not belong to their original suits (e.g. The suit of "Gras Unter" is not Gras, but Trump).
        # - Since all trumps need to be matched with other trumps, we can simply match everything by suit (no matter if trump or not).
        def true_suit(c: Card) -> int:
            return 9001 if self.game_variant.is_trump(c) else c.suit.value

        if self.game_variant.contract == GameVariant.Contract.rufspiel:
            rufsau = Card(suit=self.game_variant.ruf_suit, pip=Pip.sau)
        else:
            rufsau = None

        if len(cards_in_trick) == 0:
            # Player is leading.
            if self.game_variant.contract == GameVariant.Contract.rufspiel and true_suit(card) == self.game_variant.ruf_suit:
                # Player is playing ruf-suit.
                if rufsau in cards_in_hand:
                    # Player has the Rufsau. In that case, they are not allowed to play any card of ruf-suit unless:
                    if len(cards_in_hand) == 1:
                        # If it's the only card left.
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
            if self.game_variant.contract == GameVariant.Contract.rufspiel and true_suit(card) == self.game_variant.ruf_suit:
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
