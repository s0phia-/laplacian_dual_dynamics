from collections import defaultdict
from src.vfa.plotting.collect_plot_data import PlotAgent
import numpy as np
import random


class OfflineQLearning:
    def __init__(self, num_actions, experiences, learning_rate=0.1, discount_factor=0.99):
        self.epsilon = .1
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.experiences = experiences

    @staticmethod
    def argmax(x):
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def update(self, state):
        def find_list_of_form_ab(lists, a, b):
            """
            find state prime and reward given state and action
            """
            for sublist in lists:
                if len(sublist) >= 4 and sublist[0] == a and sublist[1] == b:
                    return sublist
            return None

        state = tuple(state)
        action = self.argmax(self.q_table[state])
        exp = find_list_of_form_ab(self.experiences, state, action)
        if exp is not None:
            _, _, reward, state_prime = exp
            action_prime = self.argmax(self.q_table[state_prime])
            td_target = reward + self.discount_factor * self.q_table[state_prime][action_prime]
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.learning_rate * td_error

    def train(self):
        for state, action, _, _ in self.experiences:
            self.update(state)

    def act(self, state):
        state = tuple(state)
        """Select an action based on the ε-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore: random action
        else:
            return self.argmax(self.q_table[state])  # Exploit: best action based on Q-values

class RunOfflineQ:
    def __init__(self, env, experiences, num_actions, agent):
        self.env = env
        self.experiences = experiences
        self.num_actions = num_actions
        self.agent = agent

    def run_loop(self):
        training_steps = 0
        return_per_step = []
        return_per_step.append(['steps', 'avg_return', 'datapoints'])
        for _ in range(20):
            self.agent.train()
            avg_return = PlotAgent(self.env, self.agent).go()
            training_steps += len(self.experiences)
            return_per_step.append([training_steps, avg_return, len(self.experiences)])
        return return_per_step
