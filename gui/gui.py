from typing import Callable

import pygame
import pygame.freetype

from game.card import new_deck, pip_scores, Pip, Suit, Card
from game.game_state import GameState, GamePhase
from gui.assets import get_card_img_path
from gui.utils import sorted_cards
from log_util import get_class_logger


class UserQuitGameException(Exception):
    # Named exception that happens when the user closes the window. This will bubble up to the controller and (likely) terminate.
    pass


class Gui:
    # The Gui is intended to operate in a single-threaded fashion.
    # Normally, we'd need to synchronize things like game_state and is_running, but in this case we don't do any sync.

    def __init__(self, game_state: GameState, resolution=(1280, 800)):
        pygame.init()

        self.logger = get_class_logger(self)

        self.game_state = game_state
        self.resolution = resolution
        self.is_running = False
        self.game_state.ev_changed.subscribe(self.on_game_state_changed)                # TODO: Also unsusbscribe at some point.

        self._screen = pygame.display.set_mode(self.resolution)                         # Display screen
        self._card_assets = {card: pygame.image.load(get_card_img_path(card)).convert() for card in new_deck()}
        self._fps_clock = pygame.time.Clock()

        # Every player has a "Card surface" onto which their cards are drawn.
        # This card surface is then rotated and translated into position.
        p_card_surf_dims = (310, 170)
        p_text_surf_dims = (200, 140)
        self._player_card_surfs = [pygame.Surface(p_card_surf_dims) for _ in range(4)]
        self._player_text_surfs = [pygame.Surface(p_text_surf_dims) for _ in range(4)]

        # Surface in the middle, containing the "cards on the table".
        self._middle_trick_surf = pygame.Surface((300, 300))

        pygame.freetype.init()
        self._font = pygame.freetype.SysFont("Courier New", 12)
        self._font.antialiased = False

        # Last click - these values survive only for one draw call. During the draw call, they are set, and afterwards read.
        self._click_pos = None
        self._clicked_card = None

        pygame.display.set_caption("AlphaSau")

    def _draw_player_cards(self):
        # Sort each player's cards before displaying. This is only for viewing in the GUI and does not affect the Player object.
        # NOTE: this is recalculated on every draw and kinda wasteful. Might want to do lazy-updating if we need UI performance.
        player_cards = [sorted_cards(cards, game_mode=self.game_state.game_mode) for cards in
                        (player.cards_in_hand for player in self.game_state.players)]

        # Draw each player's cards onto their respective card surfaces.
        for i_player in range(4):
            self._player_card_surfs[i_player].fill((0, 0, 0))
            for i_card, card in enumerate(player_cards[i_player]):
                self._player_card_surfs[i_player].blit(self._card_assets[card], (i_card*30, 0))

        # Finally, draw the card surfaces onto the board.
        self._screen.blit(self._player_card_surfs[0], (475, 600))                                       # 0: Bottom
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[1], 270), (30, 200))          # 1: Left
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[2], 180), (475, 30))          # 2: Top
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[3], 90), (1080, 200))         # 3: Right

        # Check if the user clicked on Player 0's cards.
        # In the future, let's stop using hardcoded Pixels and do some semblance of a scene graph.
        # Unfortunately, PyGame doesn't really seem to support a transform stack so let's stay with this for now. Should we move to OpenGL?
        if self._click_pos is not None and self._player_card_surfs[0].get_rect().move(475, 600).collidepoint(*self._click_pos):
            card_dims = self._card_assets[Card(Suit.schellen, Pip.sieben)].get_rect()[2:]  # Exact Width&height of any card
            rect = pygame.Rect(475, 600, *card_dims)
            n_cards = len(player_cards[0])
            clicked_card = None
            for i in reversed(range(n_cards)):
                test_rect = rect.move(i * 30, 0)           # move from right to left (front to back)
                if test_rect.collidepoint(*self._click_pos):
                    clicked_card = player_cards[0][i]
                    break

            if clicked_card:
                self._clicked_card = clicked_card
                self.logger.debug(f"Clicked on {clicked_card}")

    def _draw_current_trick_cards(self):
        # Draw the cards that are "on the table".

        self._middle_trick_surf.fill((0, 0, 0))
        if self.game_state.leading_player is None:
            return

        coords = [
            (100, 100),
            (40, 50),
            (100, 0),
            (160, 50),
            ]

        # Get the index of the leading player. The first card appears in their spot, and the rest clockwise.
        i_leader = self.game_state.players.index(self.game_state.leading_player)
        cards = self.game_state.current_trick_cards

        # Need to draw the cards in order of playing, so the first one is at the bottom.
        for i in range(4):
            i_player = (i_leader + i) % 4
            if len(cards) > i:
                self._middle_trick_surf.blit(self._card_assets[cards[i]], coords[i_player])

        self._screen.blit(self._middle_trick_surf, (480, 260))

    def _draw_player_text(self):
        decl_pid = None
        if self.game_state.game_mode is not None:
            decl_pid = self.game_state.game_mode.declaring_player_id

        txt_color = pygame.Color("#CCCCCC")
        red_color = pygame.Color("#FF0000")
        green_color = pygame.Color("#55FF55")
        for i, p in enumerate(self.game_state.players):
            score = sum(pip_scores[c.pip] for c in p.cards_in_scored_tricks)
            won = (score > 60) if i == decl_pid else (score >= 60)

            y = 0
            self._player_text_surfs[i].fill((0, 0, 0))
            self._font.render_to(self._player_text_surfs[i], (0, y), f"Player: {p.name}", fgcolor=txt_color)
            y += 20

            if i == self.game_state.i_player_dealer:
                self._font.render_to(self._player_text_surfs[i], (0, y), f"(Dealer)", fgcolor=txt_color)
                y += 20

            if p == self.game_state.leading_player:
                self._font.render_to(self._player_text_surfs[i], (0, y), f"(Leading)", fgcolor=txt_color)
                y += 20

            if i == decl_pid:
                self._font.render_to(self._player_text_surfs[i], (0, y), f"Playing a {self.game_state.game_mode}", fgcolor=red_color)
                y += 20

            self._font.render_to(self._player_text_surfs[i], (0, y), f"Score: {score}", fgcolor=green_color if won else txt_color)
            y += 20

            if self.game_state.game_phase == GamePhase.post_play and i == decl_pid:
                self._font.render_to(self._player_text_surfs[i], (0, y), "Won!" if won else "Lost!", fgcolor=green_color if won else red_color)

        self._screen.blit(self._player_text_surfs[0], (800, 600))
        self._screen.blit(self._player_text_surfs[1], (30, 520))
        self._screen.blit(self._player_text_surfs[2], (800, 30))
        self._screen.blit(self._player_text_surfs[3], (1080, 520))

    def _draw_frame(self):
        # Draws a single frame and returns control.

        self._fps_clock.tick(30)         # Limit to 30FPS
        self._screen.fill((0, 0, 0))     # Black background
        self._draw_player_cards()
        self._draw_current_trick_cards()
        self._draw_player_text()

        pygame.display.update()
        pygame.display.flip()               # Flip buffers

    def _handle_pygame_events(self):
        # Handles events from the PyGame event queue (not the GameState events!)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise UserQuitGameException

            # ESC = quit event.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise UserQuitGameException

            # Mouse button = stop drawing and return control.
            if event.type == pygame.MOUSEBUTTONUP:
                self._click_pos = pygame.mouse.get_pos()        # _draw_frame will find a card that was clicked on.

    def on_game_state_changed(self):
        # Receiving this event when we should draw an update (and wait for the user to click).

        # Wait until the user clicks.
        self._click_pos = None
        self.wait_and_draw_until(lambda: self._click_pos is not None)

        # # Example code: Wait until the user selects a card.
        # self._clicked_card = None
        # self.wait_and_draw_until(lambda: self._clicked_card is not None)

    def wait_and_draw_until(self, terminating_condition: Callable[[], bool]):
        # Runs the draw loop until the terminating condition returns true.
        while not terminating_condition():
            self._handle_pygame_events()
            self._draw_frame()
