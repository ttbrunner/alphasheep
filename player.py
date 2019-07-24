
class Player:
    def __init__(self, name):
        self.name = name
        self.cards_in_hand = []
        self.cards_in_scored_tricks = []

    def __str__(self):
        return self.name
