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

        if len(cards_in_trick) == 0:
            # Player is leading.
            if self.game_variant.contract == GameVariant.Contract.rufspiel and self.game_variant.ruf_suit == card.suit:
                # The card is of ruf-suit.
                if sum(c for c in cards_in_hand if c.suit == card.suit) >= 4:
                    # We are allowed to perform "davonlaufen".
                    return True                                         # TODO: Check rules of Davonlaufen again!
                # Player is not allowed to play the ruf-suit otherwise.
                return False

            # Leading with any card of any other suit is OK.
            return True

        # Player is not leading, so they have to match the first card.
        first_card = cards_in_trick[0]

        if self.game_variant.is_trump(first_card) and self.game_variant.is_trump(card):
            # Player is matching trump
            return True

        if first_card.suit == card.suit and not self.game_variant.is_trump(card):
            # Player is matching suit (non-trump)
            if self.game_variant.contract == GameVariant.Contract.rufspiel and card.suit == self.game_variant.ruf_suit:
                # Special case if it's the ruf-suit:
                rufsau = Card(suit=self.game_variant.ruf_suit, pip=Pip.sau)
                if card != rufsau and rufsau in cards_in_hand:
                    # Player has the ruf-sau but did not play it!
                    return False                                        # TODO: Check rules of Davonlaufen again!
            # All other cases of matching are OK.
            return True

        if self.game_variant.is_trump(first_card) and any(c for c in cards_in_hand if self.game_variant.is_trump(c)):
            # Player is not matching trump when they could.
            return False

        if any(c for c in cards_in_hand if c.suit == first_card.suit):
            # Player is not matching non-trump when they could.
            return False

        # Finally - player is not leading and not matching because they can't.
        return True
