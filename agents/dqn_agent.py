import numpy as np
from collections import deque
from typing import Iterable, List, Dict, Optional

from overrides import overrides
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam

from agents.agents import PlayerAgent
from game.card import Card, new_deck
from game.game_mode import GameMode
from log_util import get_class_logger


class DQNAgent(PlayerAgent):
    """
    First try: a cookie-cutter DQN implementation that tries to win, but can only ever see the current trick.

    Right now, it can only see its current cards and what is on the table. There is no history so in all probability, the agent
    should learn to play greedily. Or - who knows?

    In the long run, we'd like to extend the agent to deal with additional state, but for now we can experiment with this
    limited scenario.
    """

    def __init__(self, player_id: int, training: bool):
        """
        Creates a new DQNAgent.
        :param player_id: The unique id of the player (0-3).
        :param training: if True, will train during play. This usually means worse performance (because of exploration). If False,
                         then the agent will always pick the best action (according to Q-value).
        """
        super().__init__(player_id)

        self.logger = get_class_logger(self)
        self.training = training

        # In both states and actions, cards are encoded as one-hot vectors of size 32.
        # Providing indices to perform quick lookups: i->card->i
        self._cards = new_deck()
        self._card_indices = {card: i for i, card in enumerate(self._cards)}

        # See _encode_state()
        self._state_size = 32 + 3*32 + 32

        # Action space: One action for every card.
        # Naturally, most actions will be disabled because the agent doesn't have the card or is not allowed to play it.
        self._action_size = 32

        # Discount and exploration rate
        self._gamma = 0.6
        self._epsilon = 0.1

        # Experience replay buffer for minibatch learning
        self.experience_buffer = deque(maxlen=2000)

        # Remember the state and action (card) played in the previous trick, so we can can judge it once we receive feedback.
        # Also remember which actions were available - we can experiment with setting their Q to zero.
        self._prev_state = None
        self._prev_action = None
        self._prev_available_actions = None
        self._in_terminal_state = False

        # Create Q network (current state) and Target network (successor state). The networks are synced after every episode (game).
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self._align_target_model()
        self._batch_size = 32

        # Don't retrain after every single experience.
        # Retraining every time is expensive and doesn't add much information (rewards are received only at the end of the game).
        # If we wait for more experiences to accumulate before retraining, we get more fresh data before doing expensive training.
        self._retrain_every_n = 8
        self._experiences_since_last_retrain = 0

        # Memory: here are some things the agent remembers between moves. This is basically feature engineering,
        # it would be more interesting to have the agent learn these with an RNN or so!
        self._mem_cards_already_played = set()

        # For display in the GUI
        self._current_q_vals = None

    def _build_model(self):
        # Build the Q-network.
        # This is just a first shot - the game is pretty simple so intuitively I'd say the network is a bit on the large side.
        # But then again, this typically doesn't stop DL from performing well, so let's stick with it for now.

        model = Sequential()
        model.add(Dense(384, activation='relu', input_shape=(self._state_size,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        optimizer = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def _align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def _encode_state(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card]) -> np.ndarray:
        # A state contains:
        # - Cards that the player has in hand (directly observed)
        # - Cards that are in the current trick (directly observed)
        # - All cards that have been played so far (engineered feature)
        #
        # Future possibilities for features:
        # - Number of the current trick: not necessary, can be implied from len of cards_in_hand
        # - Mapping player IDs to cards to GameMode (knowing who is declaring, and then knowing THEY played a specific card)
        #   - Partially contained in the order of cards_in_trick, but needs initial info about player IDs
        # - LSTM based memory of played cards, perhaps together with player IDs
        # - Player scores, or actually a memory of all cards in all previous tricks, mapped to player IDs

        assert len(self._mem_cards_already_played) == 4 * (8-len(list(cards_in_hand)))

        state = np.zeros(shape=self._state_size, dtype=np.int32)
        offset = 0

        # 32 bools: cards in own hand (order does not matter)
        for card in cards_in_hand:
            state[offset + self._card_indices[card]] = 1
        offset += 32

        # 3x32 bools: cards in current trick before the one to be played by the agent (order is important)
        for i, card in enumerate(cards_in_trick):
            state[offset + i*32 + self._card_indices[card]] = 1
        offset += 3*32

        # 1x32 bools: cards that have already been played.
        # This is an engineered feature which could also be learned by the agent if it had some memory.
        # I'd like to try this in the future.
        for card in self._mem_cards_already_played:
            state[offset + self._card_indices[card]] = 1
        offset += 32

        assert offset == self._state_size
        return state

    def _receive_feedback(self, state, action, reward, next_state, terminated, available_actions):
        # Store the experience into the buffer and retrain the network.

        assert self.training is True
        self.experience_buffer.append((state, action, reward, next_state, terminated, available_actions))

        # Only train every n experiences (speed up training)
        self._experiences_since_last_retrain += 1
        if self._experiences_since_last_retrain < self._retrain_every_n or len(self.experience_buffer) < self._batch_size:
            return

        # Train one minibatch from the experience replay buffer.
        self._experiences_since_last_retrain = 0

        indices = np.random.choice(len(self.experience_buffer), size=self._batch_size)

        state_batch = np.empty(shape=(self._batch_size, self._state_size), dtype=np.int32)
        action_id_batch = np.empty(shape=self._batch_size, dtype=np.int32)
        reward_batch = np.empty(shape=self._batch_size, dtype=np.float32)
        next_state_batch = np.empty(shape=(self._batch_size, self._state_size), dtype=np.int32)
        terminated_batch = np.empty(shape=self._batch_size, dtype=np.bool)
        available_actions_batch = np.empty(shape=(self._batch_size, self._action_size), dtype=np.bool)

        for i, index in enumerate(indices):
            state, action, reward, next_state, terminated, available_actions = self.experience_buffer[index]
            state_batch[i, :] = state
            action_id_batch[i] = np.argmax(action)
            reward_batch[i] = reward
            next_state_batch[i, :] = next_state
            terminated_batch[i] = terminated
            available_actions_batch[i, :] = available_actions

        q_curr = self.q_network.predict(state_batch)
        q_next = self.target_network.predict(next_state_batch)

        # Terminal state: the cumulative future reward is exactly the observation.
        # Nonterminal state: the cumulative future reward is the observation + expected future reward
        # TODO on Gamma: revise notes. Would it be possible in theory to remove the discount?
        #                This should be an episodic task with rewards only in the terminal states.
        nonterminal_filter = (terminated_batch == 0)
        exp_reward = reward_batch.copy()
        exp_reward[nonterminal_filter] += self._gamma * np.amax(q_next, axis=1)[nonterminal_filter]

        # Update the Q-value for the actions that were picked. Leave the rest the same.
        q_target = q_curr.copy()
        q_target *= available_actions_batch
        q_target[np.arange(self._batch_size), action_id_batch] = exp_reward

        self.q_network.fit(state_batch, q_target, epochs=1, verbose=0)

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):
        assert cards_in_trick is not None, "Empty list is allowed, None is not."
        if self._in_terminal_state:
            raise ValueError("Agent is in terminal state. Did you start a new game? Need to call notify_new_game() first.")

        # Encode the current state.
        state = self._encode_state(cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick)

        # Save experience for training (previous action led to the current state).
        if self.training and self._prev_action is not None:
            # Right now, we provide rewards only at the end of the game.
            self._receive_feedback(state=self._prev_state, action=self._prev_action, reward=0, next_state=state,
                                   terminated=False, available_actions=self._prev_available_actions)

        # Create a mask of available actions.
        available_actions = np.zeros(self._action_size, dtype=np.bool)
        for card in cards_in_hand:
            if game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                available_actions[self._card_indices[card]] = True

        # Pick an action (a card).
        selected_card = None
        if self.training and np.random.rand() <= self._epsilon:
            # Explore: Select a random card (that is allowed).
            self._current_q_vals = np.ones(self._action_size, dtype=np.float32) / self._action_size
            cards_in_hand = list(cards_in_hand)
            np.random.shuffle(cards_in_hand)
            for card in cards_in_hand:
                if available_actions[self._card_indices[card]]:
                    selected_card = card
                    break
        else:
            # Exploit: Predict q-values for the current state and select the best action/card that is allowed.
            q_values = self.q_network.predict(state[np.newaxis, :])[0]
            self._current_q_vals = q_values
            i_best_actions = np.argsort(q_values)[::-1]
            self.logger.debug("Q values:\n" + "\n".join(f"{q_values[i]}: {self._cards[i]}" for i in i_best_actions))

            for i_action in i_best_actions:
                if available_actions[i_action]:
                    selected_card = self._cards[i_action]
                    break

        if selected_card is None:
            raise ValueError("Could not find a valid card! Please debug. Sorry.")

        # Store the state and chosen action until the next call (in which we will receive feedback)
        selected_action = np.zeros(self._action_size, dtype=np.int32)
        selected_action[self._card_indices[selected_card]] = 1

        self._prev_state = state
        self._prev_action = selected_action
        self._prev_available_actions = available_actions

        # Memory: remember cards that were played.
        self._mem_cards_already_played.update(cards_in_trick)
        self._mem_cards_already_played.add(selected_card)

        return selected_card

    @overrides
    def notify_trick_result(self, cards_in_trick: List[Card], rel_taker_id: int):
        # No aux reward for individual tricks right now.
        # But we do want to remember what players who came after us played!
        # In the future, we may also want to remember the scores of others and ourselves.

        self._mem_cards_already_played.update(cards_in_trick)

    @overrides
    def notify_game_result(self, won: bool, own_score: int, partner_score: int = None):
        # Entering the terminal state (all cards have been played and the result is announced).

        assert self._prev_action is not None and self._prev_state is not None

        if self.training:
            # In the terminal state, there are no cards
            state = self._encode_state(cards_in_hand=[], cards_in_trick=[])

            # TODO: we may want to increase reward based on total score. Something like that...?
            # reward = own_score if won else 0.
            reward = 1. if won else 0.
            self._in_terminal_state = True

            # Add feedback, sync
            self._receive_feedback(state=self._prev_state, action=self._prev_action, reward=reward, next_state=state,
                                   terminated=True, available_actions=self._prev_available_actions)
            self._align_target_model()          # The episode is over, sync the models.

    @overrides
    def notify_new_game(self):
        # Reset everything concerning the current game state.
        # Don't reset the models and experiences of course.
        # This call basically signals the begin of a new episode.

        self._prev_state = None
        self._prev_action = None
        self._prev_available_actions = None
        self._in_terminal_state = False

        self._mem_cards_already_played.clear()

    @overrides
    def internal_card_values(self) -> Optional[Dict[Card, float]]:
        return {c: self._current_q_vals[i] for i, c in enumerate(self._cards)}

    def save_weights(self, filepath, overwrite=True):
        self.logger.info(f'Saving weights to "{filepath}"...')
        self.q_network.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        self.logger.info(f'Loading weights from "{filepath}"...')
        self.q_network.load_weights(filepath)
        self._align_target_model()
