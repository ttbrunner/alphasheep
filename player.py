
class Player:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.cards_in_hand = []
        self.cards_in_scored_tricks = []

    def __str__(self):
        return "{}({})".format(self.id, self.name)
