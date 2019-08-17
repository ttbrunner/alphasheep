import os

from game.card import Card, Pip, Suit

open_tarock_dir = "assets/cards/OpenTarock/"
open_tarock_filenames = {
        Card(Suit.eichel, Pip.sau): "1.png",
        Card(Suit.gras, Pip.sau): "2.png",
        Card(Suit.herz, Pip.sau): "3.png",
        Card(Suit.schellen, Pip.sau): "4.png",
        Card(Suit.eichel, Pip.koenig): "5.png",
        Card(Suit.gras, Pip.koenig): "6.png",
        Card(Suit.herz, Pip.koenig): "7.png",
        Card(Suit.schellen, Pip.koenig): "8.png",
        Card(Suit.eichel, Pip.ober): "9.png",
        Card(Suit.gras, Pip.ober): "10.png",
        Card(Suit.herz, Pip.ober): "11.png",
        Card(Suit.schellen, Pip.ober): "12.png",
        Card(Suit.eichel, Pip.unter): "13.png",
        Card(Suit.gras, Pip.unter): "14.png",
        Card(Suit.herz, Pip.unter): "15.png",
        Card(Suit.schellen, Pip.unter): "16.png",
        Card(Suit.eichel, Pip.zehn): "17.png",
        Card(Suit.gras, Pip.zehn): "18.png",
        Card(Suit.herz, Pip.zehn): "19.png",
        Card(Suit.schellen, Pip.zehn): "20.png",
        Card(Suit.eichel, Pip.neun): "21.png",
        Card(Suit.gras, Pip.neun): "22.png",
        Card(Suit.herz, Pip.neun): "23.png",
        Card(Suit.schellen, Pip.neun): "24.png",
        Card(Suit.eichel, Pip.acht): "25.png",
        Card(Suit.gras, Pip.acht): "26.png",
        Card(Suit.herz, Pip.acht): "27.png",
        Card(Suit.schellen, Pip.acht): "28.png",
        Card(Suit.eichel, Pip.sieben): "29.png",
        Card(Suit.gras, Pip.sieben): "30.png",
        Card(Suit.herz, Pip.sieben): "31.png",
        Card(Suit.schellen, Pip.sieben): "32.png",
    }


def get_card_img_path(card: Card) -> str:
    return os.path.join(open_tarock_dir, open_tarock_filenames[card])
