from enum import Enum

from card import Suit


class GameState:

    def __init__(self, players, i_player_dealer=0):
        self.players = players
        self.i_player_dealer = i_player_dealer

        # Game mode starts with None, meaning the players have to select the game.
        self.game_mode = None


class GameType(Enum):
    rufspiel = 0,
    wenz = 1,
    solo = 2,

    # No tout/sie for now

    def __str__(self):
        return self.name


class GameMode:

    def __init__(self, game_type: GameType, ruf_suit: Suit = None, trump_suit: Suit = None):
        # Here are a couple of checks that are just for data integrity.
        # The main rule checks (e.g. which suit the player is allowed to call) must be done by the game controller during the game.

        if game_type == GameType.rufspiel:
            assert ruf_suit is not None
            assert trump_suit is None or trump_suit == Suit.herz, "Invalid trump suit for Rufspiel: {}".format(trump_suit)
            trump_suit = Suit.herz
        else:
            assert ruf_suit is None

        if game_type == GameType.solo:
            assert trump_suit is not None

        if game_type == GameType.wenz:
            assert trump_suit is None, "No Farbwenz allowed in this Wirtshaus!"

        self.game_type = game_type
        self.trump_suit = trump_suit
