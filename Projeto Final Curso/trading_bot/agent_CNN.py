import random
from collections import deque
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning."""
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

class Agent:
    """Stock Trading Bot"""

    def __init__(self, state_size, strategy="dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy
        self.state_size = state_size    # normalized previous days
        self.action_size = 3            # [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # Model config
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001    # taxa de aprendizado 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # Load or create the model
        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # Strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every
            # Target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.state_size, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions."""
        state = state.reshape(1, self.state_size, 1)  # Ajuste para entrada CNN
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # make a definite buy on the first iter

        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory."""
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                state = state.reshape(1, self.state_size, 1)         # Ajuste para entrada CNN
                next_state = next_state.reshape(1, self.state_size, 1)

                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

                q_values = self.model.predict(state, verbose=0)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                state = state.reshape(1, self.state_size, 1)
                next_state = next_state.reshape(1, self.state_size, 1)

                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])

                q_values = self.model.predict(state, verbose=0)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                state = state.reshape(1, self.state_size, 1)
                next_state = next_state.reshape(1, self.state_size, 1)

                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][np.argmax(self.model.predict(next_state, verbose=0)[0])]

                q_values = self.model.predict(state, verbose=0)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        ep = str(episode)
        nome_novo = "models/dqn.h5"
        self.model.save(nome_novo)

    def load(self):
        return load_model("models/dqn.h5", custom_objects=self.custom_objects)
