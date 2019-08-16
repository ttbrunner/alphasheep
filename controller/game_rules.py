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

        # Player is leading.
        if len(cards_in_trick) == 0:
            # Is the card of ruf-suit?
            if self.game_variant.contract == GameVariant.Contract.rufspiel and self.game_variant.ruf_suit == card.suit:
                # Can we do "davonlaufen"?
                if sum(c for c in cards_in_hand if c.suit == card.suit) >= 4:
                    raise NotImplemented("Davonlaufen not implemented yet!")
                else:
                    return False                    # If not, then it's not allowed.
            else:
                return True                         # Not ruf-suit: then anything goes.

        # Player is not leading.
        first_card = cards_in_trick[0]

        # Player is matching trump
        if self.game_variant.is_trump(first_card) and self.game_variant.is_trump(card):
            return True                         # Matching Trump

        # Player is matching suit (non-trump)
        if first_card.suit == card.suit:
            # Special case: if it's the ruf-suit and we have the Sau, must play it!
            if card.suit == self.game_variant.ruf_suit and card.pip != Pip.sau:
                if Card(suit=self.game_variant.ruf_suit, pip=Pip.sau) in cards_in_hand:
                    return False
            return True

        # Player is not matching trump. Could they?
        if self.game_variant.is_trump(first_card) and any(c for c in cards_in_hand if self.game_variant.is_trump(c)):
            return False

        # Player is not matching non-trump.
        if any(c for c in cards_in_hand if c.suit==first_card.suit):
            return False

        # Finally - player is not matching because they can't. This is OK.
        return True
