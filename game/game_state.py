from typing import List
from enum import Enum

from agents.agents import PlayerAgent
from utils import Event


class Player:
    # Game model class representing a player. The actual agent is defined by PlayerAgent.

    def __init__(self, name, agent: PlayerAgent):
        self.name = name
        self.agent = agent

        self.cards_in_hand = set()              # Unordered
        self.cards_in_scored_tricks = []        # Order of playing is important

    def __str__(self):
        return "({})".format(self.name)


class GamePhase(Enum):
    pre_deal = 0                # Nothing happening
    dealing = 1,
    bidding = 2,
    playing = 3,
    post_play = 4,              # Time to determine the winner, cleanup, post-hoc analysis.


class GameState:
    """
    GameState is the main model class of the simulator. It is intended to be reused between games.
    The controller and the GUI both have access to it, but not the Agents (of course).
    """

    def __init__(self, players: List[Player], i_player_dealer=0):
        self.players = players
        self.i_player_dealer = i_player_dealer
        for player in players:
            assert len(player.cards_in_hand) == 0 and len(player.cards_in_scored_tricks) == 0, "Can only initialize with fresh players."

        # Starting pre-deal, where the game has not been declared.
        self.game_phase = GamePhase.pre_deal
        self.game_mode = None

        # Player who plays the first card of the trick.
        self.leading_player = None

        # During the playing phase, these are the cards that are "on the table", in order of playing.
        self.current_trick_cards = []

        # Observers (such as the GUI) can subscribe to this event.
        # For now, this fires when anything (relevant) happened, like players playing cards.
        self.ev_changed = Event()

    def clear_after_game(self):
        self.game_phase = GamePhase.pre_deal
        self.game_mode = None
        self.leading_player = None
        self.current_trick_cards.clear()
        for p in self.players:
            p.cards_in_hand.clear()
            p.cards_in_scored_tricks.clear()
