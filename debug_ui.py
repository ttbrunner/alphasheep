import os

from card import new_deck
from gui.gui import Gui
from gui.deck_images import get_card_img_path


def main():
    deck = new_deck()

    img_paths = [get_card_img_path(card) for card in deck]
    for img_path in img_paths:
        assert os.path.exists(img_path)

    gui = Gui()
    gui.run()


if __name__ == '__main__':
    main()
