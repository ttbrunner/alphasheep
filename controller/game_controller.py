# Some ideas:
# - GUI kind of has to be run on the main thread, otherwise there might be problems with the event loop (or some platform problems)
# - The GUI could be some form of "game launcher" that starts the GameController, which modifies the GameState.
# - The GUI requests "nextMove" from GameController, which plays step by step, after each step returning to the caller.
# - After every "move", the GUI displays whatever is currently in the GameState.
# - In this way, the GUI acts like a debugger that can step through all the individual moves of a game.
# - Instead of the GUI runner, we can also have a "headless" runner that runs entire games without waiting (for training).
from typing import List

import numpy as np

from card import new_deck, Suit, pip_scores
from game import GameState, GamePhase, GameVariant, Player


class GameController:
    def __init__(self, players: List[Player]):
        assert len(players) == 4

        print("Initializing game.")
        print("Players:")
        for p in players:
            print("Player {} with behavior {}.".format(p, p.behavior))

        self.game_state = GameState(players)

    def run_game(self):
        def log_phase():
            print()
            print("Entering Phase: {}".format(self.game_state.game_phase))

        assert self.game_state.game_phase == GamePhase.pre_deal

        # DEALING PHASE
        # Deal random cards to players. For now, there is no laying, i.e. all 8 cards are dealt at once.
        self.game_state.game_phase = GamePhase.dealing
        log_phase()
        print("Player {} is dealing.".format(self.game_state.players[self.game_state.i_player_dealer]))
        deck = new_deck()
        np.random.shuffle(deck)
        i_deck = 0
        for p in self.game_state.players:
            p.cards_in_hand.extend(deck[i_deck:i_deck + 8])
            i_deck += 8
        self.game_state.on_changed.notify()

        # BIDDING PHASE
        # For now, no bidding allowed - a random player will do Herz-Solo.
        self.game_state.game_phase = GamePhase.bidding
        log_phase()
        i_decl = np.random.randint(4)
        g_var = GameVariant(GameVariant.Contract.suit_solo, trump_suit=Suit.herz)
        print("Game Variant: Player {} is declaring a {}!".format(self.game_state.players[i_decl], g_var))
        self.game_state.declaring_player = self.game_state.players[i_decl]
        self.game_state.game_variant = g_var
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
        print("=> Player {} {} the {}!".format(self.game_state.declaring_player, "wins" if player_win[i_decl] else "loses", g_var))
        print("Summary:")
        for i, p in enumerate(self.game_state.players):
            print("Player {} {}.".format(p, "wins" if player_win[i] else "loses"))
        self.game_state.on_changed.notify()

        # Reset to PRE-DEAL PHASE.
        self.game_state.game_phase = GamePhase.pre_deal
        log_phase()
        self.game_state.i_player_dealer = (self.game_state.i_player_dealer + 1) % 4
        self.game_state.clear_after_game()
        self.game_state.on_changed.notify()

    def _playing_phase(self):
        # Main phase of the game (trick taking).

        # Left of dealer leads the first trick.
        i_p_leader = (self.game_state.i_player_dealer + 1) % 4
        self.game_state.leading_player = self.game_state.players[i_p_leader]

        # Playing 8 tricks
        for i_trick in range(8):
            print(">>> Trick {}".format(i_trick + 1))

            # Players are playing in ascending order, starting with the leader.
            for i_p in (np.arange(4) + i_p_leader) % 4:

                # Get next card from player behavior.
                player = self.game_state.players[i_p]
                selected_card = player.behavior.play_card(player.cards_in_hand, cards_in_trick=self.game_state.current_trick_cards)

                # CHECK 1: Does the player have that card?
                # Again, this check is only for data integrity. More sophisticated logic (trying to play cards that are not available...)
                #  should be handled by the players themselves. This interface will accept only cards that exist.
                assert selected_card in player.cards_in_hand

                # CHECK 2: Do the rules allow the player to play that card?
                # TODO: Not implemented - no rule checking for now.

                print("Player {} is playing {}.".format(player, selected_card))
                player.cards_in_hand.remove(selected_card)
                self.game_state.current_trick_cards.append(selected_card)
                self.game_state.on_changed.notify()

            # Determine winner of trick.
            # TODO: Not implemented - random winner for now.
            i_win_card = np.random.randint(4)
            i_win_player = (i_p_leader + i_win_card) % 4
            win_card = self.game_state.current_trick_cards[i_win_card]
            win_player = self.game_state.players[i_win_player]
            print("Player {} wins the trick with card {}.".format(win_player, win_card))

            # Move the trick to the scored cards of the winner.
            i_p_leader = i_win_player
            self.game_state.leading_player = self.game_state.players[i_p_leader]
            win_player.cards_in_scored_tricks.extend(self.game_state.current_trick_cards)
            self.game_state.current_trick_cards.clear()
            self.game_state.on_changed.notify()

        assert sum(len(p.cards_in_scored_tricks) for p in self.game_state.players) == 32
