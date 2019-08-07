import pygame


class Gui:
    def __init__(self, resolution=(1280, 720)):
        pygame.init()

        self.resolution = resolution
        self.fps_clock = pygame.time.Clock()
        self.screen = None

    def run(self):
        self.screen = pygame.display.set_mode(self.resolution)               # Display screen
        self._draw_loop()

    def _draw_loop(self):
        # Runs until the pygame.QUIT event is received.
        running = True
        while running:

            self.fps_clock.tick(30)         # Limit to 30FPS
            self.screen.fill((0, 0, 0))     # Black background

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                # ESC = quit event
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))

            pygame.display.flip()               # Flip buffers

