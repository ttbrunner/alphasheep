from typing import Iterable, List

from agents.agents import PlayerAgent
from game.card import Card
from game.game_mode import GameMode
from utils.log_util import get_class_logger


class GUIAgent(PlayerAgent):
    """
    An agent that is part of the GUI and allows the user to play.
    For the time being, can only be used in player slot 0 (clicking on cards is only implemented for this player).

    When the GUI is __init__ed, it will find the GUIAgent in a player slot and register the callback.
    Whenever the agent has to do an action, it will run the callback - the GUI will then wait for and determine the user action.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)

        self.logger = get_class_logger(self)
        self._select_card_callback = None

    def register_gui_callback(self, select_card_callback):
        self._select_card_callback = select_card_callback

    def unregister_callback(self):
        self._select_card_callback = None

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode) -> Card:
        assert self._select_card_callback is not None, "Must first attach to a Gui!"

        # Have the user select cards until they hit something that is actually allowed :)
        reset_click = False
        while True:
            card = self._select_card_callback(reset_click)
            if game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                return card
            else:
                # Usually, the GUI caches the previous click.
                # So when a "choose" action follows a "click to continue", it's theoretically 2 clicks.
                # However, if the user clicked on a card on the FIRST click, the SECOND click will automatically choose that card.
                # That is intended behaviour, but in this case we need to reset the click cache.
                reset_click = True
                self.logger.warn(f"Cannot play selected card {card} - not allowed!")

