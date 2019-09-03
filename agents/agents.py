from abc import ABC, abstractmethod
from typing import Iterable, List, Dict, Optional
import numpy as np

from game.card import Card
from game.game_mode import GameMode


class PlayerAgent(ABC):
    """
    Abstract class for all types of agents. With "agent" here we mean the behavior of a player.
    NOTE: Only play_card() is abstract, the other methods are considered optional and have
          concrete implementations that simply do nothing. As a result, any typo or renaming might lead to
          unwanted behavior. I strongly recommend the @override decorator for the optional methods.
    """

    @abstractmethod
    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode) -> Card:
        """
        Returns the card which the agent wants to play.
        :param cards_in_hand: the cards which the player currently has in hand.
        :param cards_in_trick: cards that previous players have put into the trick (if any). If empty, then the player is leading.
        :param game_mode: the game mode that is currently being played.
        :return: A card that is contained in cards_in_hand.
        """
        pass                # Must be implemented by all agents

    def notify_trick_result(self, cards_in_trick: List[Card], rel_taker_id: int):
        """
        Notifies the agent of the result of the current trick.
        :param cards_in_trick: the four cards in the trick after all players have played theirs.
        :param rel_taker_id: the id of the player who takes the trick, relative to this player's id.
                             TODO: Replace this with a better mechanism that
                             TODO: allows the agents to map players to cards across tricks
        """
        pass                # Default implementation: do nothing

    def notify_game_result(self, won: bool, own_score: int, partner_score: int = None):
        """
        Notifies the agent of the result of the game.
        :param won: True if the player won the game.
        :param own_score: Number of points scored by the player.
        :param partner_score: Number of points scored by the player's partner. Optional: only if playing Rufspiel.
        """
        pass                # Default implementation: do nothing

    def notify_new_game(self):
        """
        Notifies the agent that a new game has started.
        """
        pass                # Default implementation: do nothing

    def internal_card_values(self) -> Optional[Dict[Card, float]]:
        """
        Gets internal values (e.g. q-values) for each Card for display.
        :return: Optional - A dictionary with a value for each card in hand.
        """
        return None         # Default implementation: no values


class RandomCardAgent(PlayerAgent):
    """
    Baseline agent. Selects a random card from its hand and plays it.
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
