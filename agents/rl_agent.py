from typing import Iterable, List

from agents.agents import PlayerAgent
from game.card import Card
from game.game_mode import GameMode


class RLAgentStub(PlayerAgent):
    """
    First try at a Reinforcement Learning agent. Mostly a stub for now.

    TODO: provide info about trick number to agents
    TODO: provide info about other player IDs to agents (so they always know who played which card)

    Anyway, we can experiment with an agent that tries to play reasonably well with the (insufficient) information they have now.
    """

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):

        raise NotImplementedError("Not done yet by a long shot.")

    def notify_game_result(self, won: bool, own_score: int, partner_score: int = None):
        # For now, this is the only reward signal (winning).
        # TODO: experiment with intermediate rewards (e.g. after each trick, maybe based on the number of points scored)
        raise NotImplementedError("Not done yet by a long shot.")