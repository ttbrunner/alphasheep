from typing import List
from enum import Enum

from card import Suit
from player_behaviour import PlayerBehavior


class Player:
    # Game model class representing a player. The actual agent is defined by PlayerBehavior.

    def __init__(self, name, behavior: PlayerBehavior):
        self.name = name
        self.behavior = behavior

        self.cards_in_hand = []
        self.cards_in_scored_tricks = []

    def __str__(self):
        return "({})".format(self.name)


class GameState:
    # GameState is the main model class of the simulator. It is intended to be reused between games (TODO: or could it be replaced?)

    def __init__(self, players: List[Player], i_player_dealer=0):
        self.players = players
        self.i_player_dealer = i_player_dealer
        for player in players:
            assert len(player.cards_in_hand) == 0 and len(player.cards_in_scored_tricks) == 0, "Can only initialize with fresh players."

        # Starting pre-deal, where the game has not been declared.
        self.game_phase = GamePhase.pre_deal
        self.game_variant = None
        self.declaring_player = None

        # During the playing phase, these are the cards that are "on the table", in order of playing.
        self.current_trick_cards = []

    def clear_after_game(self):
        self.game_phase = GamePhase.pre_deal
        self.game_variant = None
        self.declaring_player = None
        self.current_trick_cards.clear()
        for p in self.players:
            p.cards_in_hand.clear()
            p.cards_in_scored_tricks.clear()


class GamePhase(Enum):
    pre_deal = 0                # Nothing happening
    dealing = 1,
    bidding = 2,
    playing = 3,
    post_play = 4,              # Time to determine the winner, cleanup, posthoc analysis.


class GameVariant:
    class Contract(Enum):
        rufspiel = 0,
        wenz = 1,
        suit_solo = 2,
        # No tout/sie for now

        def __str__(self):
            return self.name

    def __init__(self, contract: Contract, ruf_suit: Suit = None, trump_suit: Suit = None):
        # Here are a couple of checks that are just for data integrity.
        # The main rule checks (e.g. which suit the player is allowed to call) must be done by the game controller during the game.

        if contract == self.Contract.rufspiel:
            assert ruf_suit is not None
            assert trump_suit is None or trump_suit == Suit.herz, "Invalid trump suit for Rufspiel: {}".format(trump_suit)
            trump_suit = Suit.herz
        else:
            assert ruf_suit is None

        if contract == self.Contract.suit_solo:
            assert trump_suit is not None

        if contract == self.Contract.wenz:
            assert trump_suit is None, "No Farbwenz allowed in this Wirtshaus!"

        self.contract = contract
        self.trump_suit = trump_suit
        self.ruf_suit = ruf_suit

    def __str__(self):
        if self.contract == self.Contract.suit_solo:
            return "({} solo)".format(self.trump_suit)
        elif self.contract == self.Contract.rufspiel:
            return "(rufspiel: auf die {} sau)".format(self.ruf_suit)
        else:
            return "({})".format(self.contract)

