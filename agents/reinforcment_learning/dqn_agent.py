import numpy as np
from collections import deque
from typing import Iterable, List, Dict, Optional

from overrides import overrides
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from simulator.player_agent import PlayerAgent
from simulator.card_defs import Card, new_deck
from simulator.game_mode import GameMode
from utils.log_util import get_class_logger


class DQNAgent(PlayerAgent):
    """
    A cookie-cutter DQN implementation without any sort of advanced techniques.
    """

    def __init__(self, player_id: int, config: Dict, training: bool):
        """
        Creates a new DQNAgent.
        :param player_id: The unique id of the player (0-3).
        :param config: config dict containing an agent_config node.
        :param training: If True, will train during play. This usually means worse performance (because of exploration).
                         If False, then the agent will always pick the highest-ranking valid action.
        """
        super().__init__(player_id)
        self.logger = get_class_logger(self)

        config = config["agent_config"]["dqn_agent"]
        self.config = config
        self.training = training

        # We encode cards as one-hot vectors of size 32.
        # Providing indices to perform quick lookups.
        self._id2card = new_deck()
        self._card2id = {card: i for i, card in enumerate(self._id2card)}

        # Determine length of state vector.
        state_lens = {
            "cards_in_hand": 32,
            "cards_in_trick": 3*32,
            "cards_already_played": 32
        }
        self._state_size = sum(state_lens[x] for x in config["state_contents"])

        # Action space: One action for every card.
        # Naturally, most actions will be invalid because the agent doesn't have the card or is not allowed to play it.
        self._action_size = 32

        # If True, then all unavailable actions are zeroed in the q-vector during learning. I thought this might improve training
        # speed, but it turned out to provide only a slight benefit. Incompatible with (and superseded by) allow_invalid_actions.
        self._zero_q_for_invalid_actions = config["zero_q_for_invalid_actions"]

        # If allowed, then the agent can choose an invalid card and get punished for it, while staying
        # in the same state. If not allowed, invalid actions are automatically skipped when playing.
        # See discussion in experiment_log.md
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
        # Also remember which actions were valid at that time.
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
        # NOTE: This kind of breaks the "sync networks after every game" idea, but nevertheless is working very well to speed up training.
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

        # Note that in our case, we force compile() to set the model to the old TF1 execution path. The new TF2 path is painfully slow
        #  for us, as apparently TF2 rebuilds the graph on every predict() call (in eager mode). Since our model is so small and fast, this is a
        #  huge overhead.
        model.compile(loss='mse', optimizer=Adam(lr=self.config["lr"]), experimental_run_tf_function=False)
        return model

    def _align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def _encode_state(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card]) -> np.ndarray:
        # A state contains:
        # - Cards that the player has in hand (unordered, directly observed)
        # - Cards that are in the current trick (ordered, directly observed)
        # - All cards that have been played so far (unordered, engineered feature)
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
                    state[offset + self._card2id[card]] = 1
                offset += 32

            elif comp == "cards_in_trick":
                # 3x32 bools: cards in current trick before the one to be played by the agent (order is important)
                for i, card in enumerate(cards_in_trick):
                    state[offset + i * 32 + self._card2id[card]] = 1
                offset += 3*32

            elif comp == "cards_already_played":
                # 1x32 bools: cards that have already been played.
                # This is an engineered feature which could also be learned by the agent if it had some memory.
                for card in self._mem_cards_already_played:
                    state[offset + self._card2id[card]] = 1
                offset += 32

            else:
                raise ValueError(r'Unknown state component name: "{x}"')

        assert offset == self._state_size
        return state

    def _encode_action(self, card: Card):
        action = np.zeros(self._action_size, dtype=np.int32)
        action[self._card2id[card]] = 1
        return action

    def _receive_experience(self, state, action, reward, next_state, terminated, available_actions):
        # Store the experience into the buffer and retrain the network.

        assert self.training is True
        self.experience_buffer.append((state, action, reward, next_state, terminated, available_actions))

        # Only train every n experiences (speed up training)
        self._experiences_since_last_retrain += 1
        if self._experiences_since_last_retrain < self._retrain_every_n or len(self.experience_buffer) < self._batch_size:
            return

        self._experiences_since_last_retrain = 0

        # Extract one minibatch from the experience replay buffer.
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

        # Terminal state: The cumulative future reward is exactly the observation - there are no future steps.
        # Nonterminal state: The expected cumulative future reward is the observation
        #                     + expected reward from the next state under the policy.
        #                    The amax() means that we expect the policy to pick the best action in the future.
        nonterminal_filter = (terminated_batch == 0)
        cumul_reward = reward_batch.copy()
        cumul_reward[nonterminal_filter] += self._gamma * np.amax(q_next, axis=1)[nonterminal_filter]

        # Update the Q-value for the actions that were experienced. Leave the rest the same.
        q_target = q_curr.copy()
        if self._zero_q_for_invalid_actions:            # Except, of course, for when this option is set.
            q_target *= available_actions_batch
        q_target[np.arange(self._batch_size), action_id_batch] = cumul_reward

        self.q_network.train_on_batch(state_batch, q_target)

    def play_card(self, cards_in_hand: Iterable[Card], cards_in_trick: List[Card], game_mode: GameMode):
        if self._in_terminal_state:
            raise ValueError("Agent is in terminal state. Did you start a new game? Need to call notify_new_game() first.")

        # Encode the current state.
        state = self._encode_state(cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick)

        # Did a previous action lead to this state? Save experience for training.
        if self.training and self._prev_action is not None:
            # Reward=0: We reward only the terminal state.
            self._receive_experience(state=self._prev_state, action=self._prev_action, reward=0, next_state=state,
                                     terminated=False, available_actions=self._prev_available_actions)

        # Create a mask of available actions.
        available_actions = np.zeros(self._action_size, dtype=np.bool)
        for card in cards_in_hand:
            if game_mode.is_play_allowed(card, cards_in_hand=cards_in_hand, cards_in_trick=cards_in_trick):
                available_actions[self._card2id[card]] = True

        # Pick an action (a card).
        selected_card = None
        while selected_card is None:
            # We run this in a loop, because the agent can select an invalid action and is then asked to learn and try again.

            if self.training and np.random.rand() <= self._epsilon:
                # Explore: Select a random card. For faster training, exploration only targets valid actions.
                self._current_q_vals = np.ones(self._action_size, dtype=np.float32) / self._action_size
                tmp_cards = list(cards_in_hand)
                np.random.shuffle(tmp_cards)
                selected_card = next(c for c in tmp_cards if available_actions[self._card2id[c]])
            else:
                # Exploit: Predict q-values for the current state and select the best action.
                q_values = self.q_network.predict(state[np.newaxis, :])[0]
                self._current_q_vals = q_values
                best_action_ids = np.argsort(q_values)[::-1]
                self.logger.debug("Q values:\n" + "\n".join(f"{q_values[a]}: {self._id2card[a]}" for a in best_action_ids))

                if self._allow_invalid_actions and self.training:
                    # If invalid is allowed (only during training): select the "best" action.
                    selected_card = self._id2card[best_action_ids[0]]
                    if not available_actions[best_action_ids[0]]:
                        # Did we pick an invalid move? Time for punishment!
                        # Experience: we stay in the same state, but get a negative reward.
                        self._receive_experience(state=state, action=self._encode_action(selected_card),
                                                 reward=self._invalid_action_reward,
                                                 next_state=state,
                                                 terminated=False, available_actions=available_actions)
                        selected_card = None
                else:
                    # Invalid is not allowed: pick the "best" action that is allowed.
                    selected_card = next(self._id2card[a] for a in best_action_ids if available_actions[a])

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

            # Reward is 1.0 for a game won and 0 otherwise.
            # TODO: we may want to increase reward based on total score in the future.
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
        # Report q-value per card for display / debugging.
        return {c: self._current_q_vals[i] for i, c in enumerate(self._id2card)}

    def save_weights(self, filepath, overwrite=True):
        self.logger.info(f'Saving weights to "{filepath}"...')
        self.q_network.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        self.logger.info(f'Loading weights from "{filepath}"...')
        self.q_network.load_weights(filepath)
        self._align_target_model()
