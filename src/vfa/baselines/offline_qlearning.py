from collections import defaultdict
import numpy as np
import random


class OfflineQLearning:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.99):
        self.epsilon = .1
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: np.zeros(num_actions))

    @staticmethod
    def argmax(x):
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def update(self, state, action, reward, next_state):
        best_next_action = self.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def train(self, experiences):
        for state, action, reward, next_state in experiences:
            self.update(state, action, reward, next_state)

    def act(self, state):
        """Select an action based on the ε-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore: random action
        else:
            return self.argmax(self.q_table[state])  # Exploit: best action based on Q-values
