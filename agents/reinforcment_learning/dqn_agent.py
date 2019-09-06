import numpy as np
from collections import deque
from typing import Iterable, List, Dict, Optional

from overrides import overrides
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam

from agents.agents import PlayerAgent
from game.card import Card, new_deck
from game.game_mode import GameMode
from utils.log_util import get_class_logger


class DQNAgent(PlayerAgent):
    """
    First try: a cookie-cutter DQN implementation.
    """

    def __init__(self, player_id: int, config: Dict, training: bool):
        """
        Creates a new DQNAgent.
        :param player_id: The unique id of the player (0-3).
        :param config: the config dict (root).
        :param training: if True, will train during play. This usually means worse performance (because of exploration). If False,
                         then the agent will always pick the best action (according to Q-value).
        """
        super().__init__(player_id)

        self.logger = get_class_logger(self)
        config = config["agent_config"]["dqn_agent"]
        self.config = config
        self.training = training

        # In both states and actions, cards are encoded as one-hot vectors of size 32.
        # Providing indices to perform quick lookups: i->card->i
        self._cards = new_deck()
        self._card_indices = {card: i for i, card in enumerate(self._cards)}

        # Determine length of state vector.
        state_lens = {
            "cards_in_hand": 32,
            "cards_in_trick": 3*32,
            "cards_already_played": 32
        }
        self._state_size = sum(state_lens[x] for x in config["state_contents"])

        # Action space: One action for every card.
        # Naturally, most actions will be disabled because the agent doesn't have the card or is not allowed to play it.
        self._action_size = 32

        # If True, then all unavailable actions are zeroed in the q-vector during learning. In the original
        # setup, this equals "enter a terminal state where the game is lost". I thought this might improve training
        # speed, but not sure right now.
        self._zero_q_for_invalid_actions = config["zero_q_for_invalid_actions"]

        # If allowed, then the agent can "try" to play an invalid card and gets punished for it, while staying
        # in the same state. If not allowed, invalid actions are automatically skipped when playing.
        # See discussion in observations.md
        self._allow_invalid_actions = config["allow_invalid_actions"]
        self._invalid_action_reward = config["invalid_action_reward"]
        if self._allow_invalid_actions and self._zero_q_for_invalid_actions:
            raise ValueError("allow_invalid_actions and zero_q_for_invalid_actions are mutually exclusive.")

        # Discount and exploration rate
        self._gamma = config["gamma"]
        self._epsilon = config["epsilon"]

        # Experience replay buffer for minibatch learning
        self.experience_buffer = deque(maxlen=config["experience_buffer_len"])

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
        self._batch_size = config["batch_size"]

        # Don't retrain after every single experience.
        # Retraining every time is expensive and doesn't add much information (rewards are received only at the end of the game).
        # If we wait for more experiences to accumulate before retraining, we get more fresh data before doing expensive training.
        self._retrain_every_n = config["retrain_every"]
        self._experiences_since_last_retrain = 0

        # Memory: here are some things the agent remembers between moves. This is basically feature engineering,
        # it would be more interesting to have the agent learn these with an RNN or so!
        self._mem_cards_already_played = set()

        # For display in the GUI
        self._current_q_vals = None

    def _build_model(self):
        # Build the Q-network.

        model = Sequential()
        model.add(Input(shape=(self._state_size,)))
        for i, neurons in enumerate(self.config["model_neurons"]):
            model.add(Dense(neurons, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        optimizer = Adam(lr=self.config["lr"])
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

        for comp in self.config["state_contents"]:
            if comp == "cards_in_hand":
                # 32 bools: cards in own hand (order does not matter)
                for card in cards_in_hand:
                    state[offset + self._card_indices[card]] = 1
                offset += 32

            elif comp == "cards_in_trick":
                # 3x32 bools: cards in current trick before the one to be played by the agent (order is important)
                for i, card in enumerate(cards_in_trick):
                    state[offset + i*32 + self._card_indices[card]] = 1
                offset += 3*32

            elif comp == "cards_already_played":
                # 1x32 bools: cards that have already been played.
                # This is an engineered feature which could also be learned by the agent if it had some memory.
                # I'd like to try this in the future.
                for card in self._mem_cards_already_played:
                    state[offset + self._card_indices[card]] = 1
                offset += 32

            else:
                raise ValueError(r'Unknown state component name: "{x}"')

        assert offset == self._state_size
        return state

    def _encode_action(self, card: Card):
        action = np.zeros(self._action_size, dtype=np.int32)
        action[self._card_indices[card]] = 1
        return action

    def _receive_experience(self, state, action, reward, next_state, terminated, available_actions):
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
        if self._zero_q_for_invalid_actions:
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
            self._receive_experience(state=self._prev_state, action=self._prev_action, reward=0, next_state=state,
                                     terminated=False, available_actions=self._prev_available_actions)

        # Create a mask of available actions.
        available_actions = np.zeros(self._action_size, dtype=np.bool)
        for card in cards_in_hand:
            if game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                available_actions[self._card_indices[card]] = True

        # Pick an action (a card).
        selected_card = None
        while selected_card is None:
            # We run this in a loop, because the agent can select an invalid action and is then asked to try again.

            if self.training and np.random.rand() <= self._epsilon:
                # Explore: Select a random card - only exploring allowed actions for now.
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

                if self._allow_invalid_actions and self.training:
                    # If invalid is allowed (only during training): select the "best" action.
                    selected_card = self._cards[i_best_actions[0]]
                    if not available_actions[i_best_actions[0]]:
                        # Did we pick an invalid move? Time for punishment!
                        # Experience: we stay in the same state, but get a negative reward.
                        self._receive_experience(state=state, action=self._encode_action(selected_card),
                                                 reward=self._invalid_action_reward,
                                                 next_state=state,
                                                 terminated=False, available_actions=available_actions)
                        selected_card = None
                else:
                    # Only valid are allowed: pick the "best" action that is allowed.
                    for i_action in i_best_actions:
                        if available_actions[i_action]:
                            selected_card = self._cards[i_action]
                            break

        if selected_card is None:
            raise ValueError("Could not find a valid card! Please debug. Sorry.")

        # Store the state and chosen action until the next call (in which we will receive feedback)
        self._prev_state = state
        self._prev_action = self._encode_action(selected_card)
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
            self._receive_experience(state=self._prev_state, action=self._prev_action, reward=reward, next_state=state,
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
