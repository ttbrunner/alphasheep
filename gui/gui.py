import pygame

from card import new_deck, Card, Pip, Suit
from gui.deck_images import get_card_img_path


class Gui:
    def __init__(self, players, resolution=(1280, 800)):
        pygame.init()

        self.resolution = resolution
        self._fps_clock = pygame.time.Clock()
        self._screen = None
        self._card_assets = None

        # Every player has a "Card surface" onto which their cards are drawn.
        # This card surface is then rotated and translated into position.
        card_surf_dims = (400, 200)
        self._player_card_surfs = [pygame.Surface(card_surf_dims) for _ in range(4)]

        self.players = players

        print("yolo")

    def _load_assets(self):
        self._card_assets = {card: pygame.image.load(get_card_img_path(card)).convert() for card in new_deck()}

    def _draw_player_cards(self):
        # Draw each player's cards onto their card surfaces.
        for i_player in range(4):
            for i_card, card in enumerate(self.players[i_player].cards_in_hand):
                self._player_card_surfs[i_player].blit(self._card_assets[card], (i_card*30, 0))

        # Finally, draw the card surfaces onto the board.
        self._screen.blit(self._player_card_surfs[0], (475, 600))
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[1], 90), (30, 170))
        self._screen.blit(self._player_card_surfs[2], (475, 30))
        self._screen.blit(pygame.transform.rotate(self._player_card_surfs[3], 90), (1080, 170))

    def _draw_loop(self):
        # Runs until the pygame.QUIT event is received.
        running = True
        while running:

            self._fps_clock.tick(30)         # Limit to 30FPS
            self._screen.fill((0, 0, 0))     # Black background
            self._draw_player_cards()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                # ESC = quit event
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))

            pygame.display.flip()               # Flip buffers

    def run(self):
        self._screen = pygame.display.set_mode(self.resolution)               # Display screen
        pygame.display.set_caption("AlphaSau")
        self._load_assets()
        self._draw_loop()
