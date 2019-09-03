from typing import List

import numpy as np

from controller.dealing_behavior import DealFairly, DealingBehavior
from game.card import Suit, pip_scores
from game.game_mode import GameMode, GameContract
from game.game_state import Player, GameState, GamePhase
from log_util import get_class_logger


class GameController:
    def __init__(self, players: List[Player], i_player_dealer=0,
                 dealing_behavior: DealingBehavior = DealFairly(), forced_game_mode: GameMode = None):
        """
        Creates a GameController and, together with it, a GameState. Should be reused - run run_game() in order to simulate a single game.
        :param players: the players, along with their agents.
        :param i_player_dealer: The player who is the dealer at start (i+1 is the player who will lead in the first game).
        :param dealing_behavior: Optional - the dealing behaviour. Default = fair
        :param forced_game_mode: Optional - if not None, players cannot bid, but every game is always the provided mode.
        """
        assert len(players) == 4

        self.logger = get_class_logger(self)
        self.logger.debug("Initializing game.")
        self.logger.debug("Players:")
        for p in players:
            self.logger.debug("Player {} with behavior {}.".format(p, p.agent))

        self.game_state = GameState(players, i_player_dealer=i_player_dealer)
        self.dealing_behavior = dealing_behavior
        self.forced_game_mode = forced_game_mode
        assert forced_game_mode is None or forced_game_mode.declaring_player_id is not None, "Must provide a specific player."

    def run_game(self) -> List[bool]:
        """
        Runs a single game (and shifts the dealing player clockwise). Can be called multiple times.
        :returns a list of 4 bools, indicating which player(s) won the game.
        """

        def log_phase():
            self.logger.debug("===== Entering Phase: {} =====".format(self.game_state.game_phase))

        assert self.game_state.game_phase == GamePhase.pre_deal

        for p in self.game_state.players:
            p.agent.notify_new_game()

        # DEALING PHASE
        self.game_state.game_phase = GamePhase.dealing
        log_phase()
        self.logger.debug("Player {} is dealing.".format(self.game_state.players[self.game_state.i_player_dealer]))
        hands = self.dealing_behavior.deal_hands()
        for i, p in enumerate(self.game_state.players):
            p.cards_in_hand = hands[i]
        self.game_state.ev_changed.notify()

        # BIDDING PHASE
        # Choose the game mode and declaring player.
        self.game_state.game_phase = GamePhase.bidding
        log_phase()
        if self.forced_game_mode is not None:
            # We have been instructed to only play this game.
            game_mode = self.forced_game_mode
        else:
            # Free choice - for now, randomly select somebody to play a Herz Solo.
            # TODO: allow agents to bid & declare on their own
            game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=np.random.randint(4))
        i_decl = game_mode.declaring_player_id
        self.logger.debug("Game Variant: Player {} is declaring a {}!".format(self.game_state.players[i_decl], game_mode))
        self.game_state.game_mode = game_mode
        self.game_state.ev_changed.notify()

        # PLAYING PHASE
        self.game_state.game_phase = GamePhase.playing
        log_phase()
        self._playing_phase()

        # POST-GAME PHASE
        # Count score and determine winner.
        # TODO: For now, always scoring a solo.
        self.game_state.game_phase = GamePhase.post_play
        log_phase()

        player_scores = [sum(pip_scores[c.pip] for c in p.cards_in_scored_tricks) for p in self.game_state.players]
        for i, p in enumerate(self.game_state.players):
            self.logger.debug("Player {} has score {}.".format(p, player_scores[i]))
        if player_scores[i_decl] > 60:
            player_win = [i == i_decl for i in range(4)]
        else:
            player_win = [i != i_decl for i in range(4)]
        self.logger.debug("=> Player {} {} the {}!".format(self.game_state.players[i_decl], "wins" if player_win[i_decl] else "loses",
                                                           game_mode))

        self.logger.debug("Summary:")
        for i, p in enumerate(self.game_state.players):
            self.logger.debug("Player {} {}.".format(p, "wins" if player_win[i] else "loses"))
            p.agent.notify_game_result(player_win[i], own_score=player_scores[i])
        self.game_state.ev_changed.notify()

        # Reset to PRE-DEAL PHASE.
        self.game_state.game_phase = GamePhase.pre_deal
        log_phase()
        self.game_state.clear_after_game()
        self.game_state.i_player_dealer = (self.game_state.i_player_dealer + 1) % 4
        self.game_state.ev_changed.notify()

        return player_win

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
            self.logger.debug("-- Trick {} --".format(i_trick + 1))

            # Players are playing in ascending order, starting with the leader.
            for i_p in (np.arange(4) + i_p_leader) % 4:

                # Get next card from player agent.
                player = game_state.players[i_p]
                selected_card = player.agent.play_card(player.cards_in_hand,
                                                       cards_in_trick=game_state.current_trick_cards,
                                                       game_mode=game_mode)

                # CHECK 1: Does the player have that card?
                # Again, this check is only for data integrity. More sophisticated logic (trying to play cards that are not available...)
                #  should be handled by the players themselves. This interface will accept only cards that exist.
                assert selected_card in player.cards_in_hand, f"{player} does not have {selected_card}!"

                # CHECK 2: Do the rules allow the player to play that card?
                if not game_mode.is_play_allowed(selected_card,
                                                 cards_in_hand=player.cards_in_hand,
                                                 cards_in_trick=game_state.current_trick_cards):
                    raise ValueError("Player {} tried to play {}, but it's not allowed!".format(player, selected_card))

                self.logger.debug("Player {} is playing {}.".format(player, selected_card))
                player.cards_in_hand.remove(selected_card)
                game_state.current_trick_cards.append(selected_card)
                game_state.ev_changed.notify()

            # Determine winner of trick.
            i_win_card = game_mode.get_trick_winner(game_state.current_trick_cards)
            i_win_player = (i_p_leader + i_win_card) % 4
            win_card = game_state.current_trick_cards[i_win_card]
            win_player = game_state.players[i_win_player]
            self.logger.debug("Player {} wins the trick with card {}.".format(win_player, win_card))
            for i, p in enumerate(self.game_state.players):
                p.agent.notify_trick_result(game_state.current_trick_cards, rel_taker_id=i-i_win_player)

            # Move the trick to the scored cards of the winner.
            i_p_leader = i_win_player
            game_state.leading_player = game_state.players[i_p_leader]
            win_player.cards_in_scored_tricks.extend(game_state.current_trick_cards)
            game_state.current_trick_cards.clear()
            game_state.ev_changed.notify()

        assert sum(len(p.cards_in_scored_tricks) for p in game_state.players) == 32
