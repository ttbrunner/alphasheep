import pygame

from card import new_deck, Card, Pip, Suit
from game import GameState
from gui.deck_images import get_card_img_path
from gui.utils import sorted_cards


class Gui:
    # The Gui is intended to operate in a single-threaded fashion.
    # Normally, we'd need to synchronize things like game_state and is_running, but in this case we don't do any sync.
    #

    def __init__(self, game_state: GameState, resolution=(1280, 800)):
        pygame.init()

        self.game_state = game_state
        self.resolution = resolution
        self.is_running = False

        self._fps_clock = pygame.time.Clock()
        self._screen = None
        self._card_assets = None

        # Every player has a "Card surface" onto which their cards are drawn.
        # This card surface is then rotated and translated into position.
        card_surf_dims = (310, 170)
        self._player_card_surfs = [pygame.Surface(card_surf_dims) for _ in range(4)]

    def _load_assets(self):
        self._card_assets = {card: pygame.image.load(get_card_img_path(card)).convert() for card in new_deck()}

    def _draw_player_cards(self):
        # Sort each player's cards before displaying. This is only for viewing in the GUI and does not affect the true card list.
        # NOTE: this is recalculated on every draw and kinda wasteful. Might want to do lazy-updating if we need UI performance.
        player_cards = [sorted_cards(cards, game_mode=self.game_state.game_mode) for cards in
                        (player.cards_in_hand for player in self.game_state.players)]

        # Draw each player's cards onto their respective card surfaces.
        for i_player in range(4):
            for i_card, card in enumerate(player_cards[i_player]):
                self._player_card_surfs[i_player].blit(self._card_assets[card], (i_card*30, 0))

        # Finally, draw the card surfaces onto the board.
        self._screen.blit(self._player_card_surfs[0], (475, 600))
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[1], 270), (30, 220))
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[2], 180), (475, 30))
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[3], 90), (1080, 170))

    def _draw_loop(self):
        # Runs until the pygame.QUIT event is received.
        while True:

            self._fps_clock.tick(30)         # Limit to 30FPS
            self._screen.fill((0, 0, 0))     # Black background
            self._draw_player_cards()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                # ESC = quit event
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))

            pygame.display.flip()               # Flip buffers

    def run(self):
        self.is_running = True
        self._screen = pygame.display.set_mode(self.resolution)               # Display screen
        pygame.display.set_caption("AlphaSau")
        self._load_assets()
        self._draw_loop()
        self.is_running = False
