import pygame

from game.card import new_deck
from game.game_state import GameState
from gui.assets import get_card_img_path
from gui.utils import sorted_cards


class UserQuitGameException(Exception):
    # Named exception that happens when the user closes the window. This will bubble up to the controller and (likely) terminate.
    pass


class Gui:
    # The Gui is intended to operate in a single-threaded fashion.
    # Normally, we'd need to synchronize things like game_state and is_running, but in this case we don't do any sync.

    def __init__(self, game_state: GameState, resolution=(1280, 800)):
        pygame.init()

        self.game_state = game_state
        self.resolution = resolution
        self.is_running = False
        self.game_state.ev_changed.subscribe(self.on_game_state_changed)                # TODO: This should be cleaned up.

        self._screen = pygame.display.set_mode(self.resolution)                         # Display screen
        self._card_assets = {card: pygame.image.load(get_card_img_path(card)).convert() for card in new_deck()}
        self._fps_clock = pygame.time.Clock()
        self._waiting_for_click = False

        # Every player has a "Card surface" onto which their cards are drawn.
        # This card surface is then rotated and translated into position.
        card_surf_dims = (310, 170)
        self._player_card_surfs = [pygame.Surface(card_surf_dims) for _ in range(4)]

        # Surface in the middle, containing the "cards on the table".
        self._middle_trick_surf = pygame.Surface((300, 300))

        pygame.display.set_caption("AlphaSau")

    def _draw_player_cards(self):
        # Sort each player's cards before displaying. This is only for viewing in the GUI and does not affect the true card list.
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
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[1], 270), (30, 220))          # 1: Left
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[2], 180), (475, 30))          # 2: Top
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[3], 90), (1080, 170))         # 3: Right

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

    def _draw_frame(self):
        # Draws a single frame and returns control.

        self._fps_clock.tick(30)         # Limit to 30FPS
        self._screen.fill((0, 0, 0))     # Black background
        self._draw_player_cards()
        self._draw_current_trick_cards()

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
                self._waiting_for_click = False

    def on_game_state_changed(self):
        # Receiving this event when we should draw an update (and maybe pause).

        # Run draw loop until a click was received.
        self._waiting_for_click = True
        while self._waiting_for_click:
            self._handle_pygame_events()
            self._draw_frame()

        # pygame.time.wait(1000)
