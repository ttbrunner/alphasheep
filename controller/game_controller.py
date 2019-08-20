# Some ideas:
# - GUI kind of has to be run on the main thread, otherwise there might be problems with the event loop (or some platform problems)
# - The GUI could be some form of "game launcher" that starts the GameController, which modifies the GameState.
# - The GUI requests "nextMove" from GameController, which plays step by step, after each step returning to the caller.
# - After every "move", the GUI displays whatever is currently in the GameState.
# - In this way, the GUI acts like a debugger that can step through all the individual moves of a game.
# - Instead of the GUI runner, we can also have a "headless" runner that runs entire games without waiting (for training).
from typing import List

import numpy as np

from game.card import new_deck, Suit, pip_scores, Pip, Card
from game.game_mode import GameMode, GameContract
from game.game_state import Player, GameState, GamePhase


class GameController:
    def __init__(self, players: List[Player]):
        """
        Creates a GameController and, together with it, a GameState. Should be reused - run run_game() in order to simulate a single game.
        :param players: the players, along with their agents.
        """
        assert len(players) == 4

        print("Initializing game.")
        print("Players:")
        for p in players:
            print("Player {} with behavior {}.".format(p, p.agent))

        self.game_state = GameState(players)

    def run_game(self):
        """
        Runs a single game (and shifts the dealing player clockwise). Can be called multiple times.
        """

        def log_phase():
            print()
            print("Entering Phase: {}".format(self.game_state.game_phase))

        assert self.game_state.game_phase == GamePhase.pre_deal

        for p in self.game_state.players:
            p.agent.notify_new_game()

        # DEALING PHASE
        # Deal random cards to players. For now, there is no laying, i.e. all 8 cards are dealt at once.
        # Simplification: we keep shuffling until somebody can play a Herz Solo.
        # TODO: deal normally and allow agents to bid & declare on their own
        self.game_state.game_phase = GamePhase.dealing
        log_phase()
        print("Player {} is dealing.".format(self.game_state.players[self.game_state.i_player_dealer]))
        deck = new_deck()
        game_mode = None
        while game_mode is None:
            np.random.shuffle(deck)
            i_deck = 0
            for p in self.game_state.players:
                p.cards_in_hand.clear()
                p.cards_in_hand.extend(deck[i_deck:i_deck + 8])
                i_deck += 8
            game_mode = self._pick_best_game()
        self.game_state.on_changed.notify()

        # BIDDING PHASE
        # For now, no bidding allowed - we made sure somebody can play a Herz Solo, and that they will!
        # Like this, every game is always a Herz Solo, although not always the same player.
        # TODO: allow agents to bid & declare on their own
        self.game_state.game_phase = GamePhase.bidding
        log_phase()
        i_decl = game_mode.declaring_player_id
        print("Game Variant: Player {} is declaring a {}!".format(self.game_state.players[i_decl], game_mode))
        self.game_state.game_mode = game_mode
        self.game_state.on_changed.notify()

        # PLAYING PHASE
        self.game_state.game_phase = GamePhase.playing
        log_phase()
        self._playing_phase()

        # POST-GAME PHASE: Count score and determine winner.
        self.game_state.game_phase = GamePhase.post_play
        log_phase()
        player_scores = [sum(pip_scores[c.pip] for c in p.cards_in_scored_tricks) for p in self.game_state.players]
        for i, p in enumerate(self.game_state.players):
            print("Player {} has score {}.".format(p, player_scores[i]))

        # TODO: For now, always scoring a solo.
        if player_scores[i_decl] > 60:
            player_win = [i == i_decl for i in range(4)]
        else:
            player_win = [i != i_decl for i in range(4)]
        print("=> Player {} {} the {}!".format(self.game_state.players[i_decl], "wins" if player_win[i_decl] else "loses", game_mode))
        print("Summary:")
        for i, p in enumerate(self.game_state.players):
            print("Player {} {}.".format(p, "wins" if player_win[i] else "loses"))
            p.agent.notify_game_result(player_win[i], own_score=player_scores[i])
        self.game_state.on_changed.notify()

        # Reset to PRE-DEAL PHASE.
        self.game_state.game_phase = GamePhase.pre_deal
        log_phase()
        self.game_state.clear_after_game()
        self.game_state.i_player_dealer = (self.game_state.i_player_dealer + 1) % 4
        self.game_state.on_changed.notify()

    def _playing_phase(self):
        # Main phase of the game (trick taking).

        # Some shortcuts
        game_state = self.game_state
        game_mode = self.game_state.game_mode

        # Left of dealer leads the first trick.
        i_p_leader = (game_state.i_player_dealer + 1) % 4
        game_state.leading_player = game_state.players[i_p_leader]

        # Playing 8 tricks
        for i_trick in range(8):
            print(">>> Trick {}".format(i_trick + 1))

            # Players are playing in ascending order, starting with the leader.
            for i_p in (np.arange(4) + i_p_leader) % 4:

                # Get next card from player agent.
                player = game_state.players[i_p]
                selected_card = player.agent.play_card(player.cards_in_hand, cards_in_trick=game_state.current_trick_cards, game_mode=game_mode)

                # CHECK 1: Does the player have that card?
                # Again, this check is only for data integrity. More sophisticated logic (trying to play cards that are not available...)
                #  should be handled by the players themselves. This interface will accept only cards that exist.
                assert selected_card in player.cards_in_hand

                # CHECK 2: Do the rules allow the player to play that card?
                if not game_mode.is_play_allowed(selected_card, cards_in_hand=player.cards_in_hand, cards_in_trick=game_state.current_trick_cards):
                    raise print("Player {} tried to play {}, but it's not allowed!".format(player, selected_card))

                print("Player {} is playing {}.".format(player, selected_card))
                player.cards_in_hand.remove(selected_card)
                game_state.current_trick_cards.append(selected_card)
                game_state.on_changed.notify()

            # Determine winner of trick.
            i_win_card = game_mode.get_trick_winner(game_state.current_trick_cards)
            i_win_player = (i_p_leader + i_win_card) % 4
            win_card = game_state.current_trick_cards[i_win_card]
            win_player = game_state.players[i_win_player]
            print("Player {} wins the trick with card {}.".format(win_player, win_card))

            # Move the trick to the scored cards of the winner.
            i_p_leader = i_win_player
            game_state.leading_player = game_state.players[i_p_leader]
            win_player.cards_in_scored_tricks.extend(game_state.current_trick_cards)
            game_state.current_trick_cards.clear()
            game_state.on_changed.notify()

        assert sum(len(p.cards_in_scored_tricks) for p in game_state.players) == 32

    def _pick_best_game(self):
        # Depending on the cards of all players, pick a game to play.
        # For now, the controller decides the game mode and simply condemns the players to play it. In this version, only Herz-solo is supported.
        # In this way, we will re-shuffle until we decide a Herz-Solo is playable. Thus all games can be reasonable Herz-Solos.
        # TODO: Allow bidding for agents
        # TODO: Could pick other modes as well.

        # Check if it's reasonably to have somebody play a Herz Solo.
        game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=None)

        def can_play_suit_solo(p):
            # Needs 6 trumps and either good Obers or lots of Unters.
            if sum(1 for c in p.cards_in_hand if game_mode.is_trump(c)) >= 6:
                if sum(1 for c in p.cards_in_hand if c.pip == Pip.ober) >= 2:
                    return True
                elif sum(1 for c in p.cards_in_hand if c.pip == Pip.unter) >= 3:
                    return True
                elif Card(Suit.eichel, Pip.ober) in p.cards_in_hand:
                    return True
            return False

        for i_p, player in enumerate(self.game_state.players):
            if can_play_suit_solo(player):
                game_mode.declaring_player_id = i_p
                return game_mode

        return None                         # Nobody is playing anything



