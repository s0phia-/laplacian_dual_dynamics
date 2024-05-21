import numpy as np
import random
from collections import defaultdict
from src.tools.random import set_random_number_generator


class Lstdq:
    """
    Least squares temporal difference Q learning
    """

    def __init__(self,
                 num_actions: int,
                 seed: int = 0,
                 policy: list = [],
                 eigenvectors: np.array = [],
                 state_map: list = [],
                 source_of_samples: list = [],
                 random_number_generator: random.Random = None,):

        # hyperparams
        self.gamma = .95  # discount factor
        self.epsilon = 0.1

        # get attrs
        self.num_actions = num_actions
        self.samples = source_of_samples  # called D in LSPI paper
        self.policy_matrix = policy
        self.state_map = state_map
        self.eigenvectors = eigenvectors

        # set up
        if eigenvectors is not []:
            self.len_evs = eigenvectors.shape[1]
            self.matrix_A = np.zeros([self.len_evs * self.num_actions, self.len_evs * self.num_actions])
            self.vector_b = np.zeros([self.len_evs * self.num_actions])
            self.eigenvectors = defaultdict(lambda: np.ones(self.len_evs))
            for n in range(eigenvectors.shape[0]):
                self.eigenvectors[n] = eigenvectors[n]
        set_random_number_generator(self, random_number_generator, seed)

    def act(self, state):
        """
        return the action to take, given the state
        """

        if np.random.rand() < self.epsilon or not self.eigenvectors:
            return self.random_number_generator.randint(0, self.num_actions-1)
        else:
            state = tuple(state.get("agent"))
            state = self.eigenvectors[self.state_map[state]]

            return self.greedy_policy(state)

    @staticmethod
    def random_tiebreak_argmax(x):
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def apply_bf(self, state_evs, action):
        sa_bf = np.zeros([self.num_actions, self.len_evs])
        sa_bf[action] = state_evs
        return sa_bf.flatten()

    def greedy_policy(self, state):
        q_values = np.matmul(self.policy_matrix, state)
        return self.random_tiebreak_argmax(q_values)

    def find_a_and_b(self):
        def get_formatted_state(x, action=None):
            x = tuple(x.get("agent"))
            x = self.state_map[x]
            x = self.eigenvectors[x]
            if action == None:
                action = self.greedy_policy(x)
            x = self.apply_bf(x, action)
            return x

        for d_i in self.samples:
            state, action, reward, state_prime = d_i
            state_action = get_formatted_state(state, action)
            state_action_prime = get_formatted_state(state_prime)

            x = state_action - self.gamma * state_action_prime
            self.matrix_A += np.matmul(state_action.reshape([state_action.shape[0], 1]), x.reshape([1, x.shape[0]]))
            self.vector_b += state_action * reward

    def fit(self):
        self.find_a_and_b()
        policy = np.matmul(np.linalg.inv(self.matrix_A), self.vector_b)
        return policy.reshape([self.num_actions, self.len_evs])


class LspiAgent:
    def __init__(self,
                 eigenvectors,
                 state_map,
                 env,
                 actions,
                 max_samples: int = 10 ** 6,
                 source_of_samples: list = [],
                 seed=0):
        self.eigenvectors = eigenvectors  # array of eigenvectors
        self.state_map = state_map  # mapping from states to eigenvector index
        self.source_of_samples = source_of_samples
        self.max_samples = max_samples
        self.len_evs = eigenvectors.shape[1]
        self.num_actions = actions.n
        self.policy = np.random.rand(self.num_actions, self.len_evs)
        self.action_space = actions
        self.model = Lstdq
        self.env = env
        self.seed = seed
        np.random.seed = seed

    @staticmethod
    def random_tiebreak_argmax(x):
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def learn(self, stopping_criteria, max_out=1000):
        diff = stopping_criteria + 1
        diff_list = []
        training_steps = 0
        return_per_step = []
        for i in range(max_out):
            if diff > stopping_criteria or diff not in diff_list:
                w_old = self.policy

                agent = self.model(num_actions=self.num_actions,
                                   seed=self.seed,
                                   policy=w_old,
                                   eigenvectors=self.eigenvectors,
                                   state_map=self.state_map,
                                   source_of_samples=self.source_of_samples)
                w_new = agent.fit()
                self.policy = w_new
                diff = round(np.linalg.norm(w_old - w_new), 2)
                diff_list.append(diff)
                avg_return = PlotAgent(env=self.env, initialised_agent=Lstdq(num_actions=4,
                                                                             policy=self.policy,
                                                                             eigenvectors=self.eigenvectors,
                                                                             state_map=self.state_map,
                                                                             source_of_samples=None)).go()
                training_steps += len(self.source_of_samples)
                print(diff, avg_return, training_steps)
                return_per_step.append([training_steps, avg_return])
            else:
                break
        return w_new, return_per_step

    def run(self, experience, stopping_criteria):
        # self.collect_experience(episodes=episodes, env=env, max_episode_length=max_episode_length)
        self.learn(stopping_criteria=stopping_criteria, source_of_samples=experience)


class PlotAgent:
    def __init__(self, env, initialised_agent, agents_to_avg=50):
        self.toplot = []
        self.env = env
        self.agent = initialised_agent
        self.i = agents_to_avg
    def go(self):
        for _ in range(self.i):
            state, _ = self.env.reset()
            step_count = 0
            cum_reward = 0
            terminated = False
            train_step = 0
            while not terminated and step_count < 10 ** 4:
                # get action
                action = self.agent.act(state)
                # step
                state, reward, terminated, truncated, _ = self.env.step(action)
                # add reward cumulatively
                cum_reward += reward
                train_step += 1
                step_count += 1
            self.toplot.append(cum_reward)
        return np.mean(self.toplot)
