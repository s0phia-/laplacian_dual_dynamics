import numpy as np
import random


class Lstdq:
    """
    Least squares temporal difference Q learning
    """

    def __init__(self, num_actions, policy, eigenvectors, source_of_samples):
        self.len_evs = len(eigenvectors[0])
        self.num_actions = num_actions
        self.gamma = .95  # discount factor
        self.matrix_A = np.zeros([self.len_evs * self.num_actions, self.len_evs * self.num_actions])
        self.vector_b = np.zeros([self.len_evs * self.num_actions])
        self.samples = source_of_samples  # called D in LSPI paper
        self.policy_matrix = policy
        self.ev_map = eigenvectors

    @staticmethod
    def random_tiebreak_argmax(x):
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def apply_bf(self, state, action):
        sa_bf = np.zeros([self.num_actions, self.len_evs])
        sa_bf[action] = state
        return sa_bf.flatten()

    def greedy_policy(self, state):
        q_values = np.matmul(self.policy_matrix, state)
        return self.random_tiebreak_argmax(q_values)

    def find_a_and_b(self):
        for d_i in self.samples:
            state, action, reward, state_prime = d_i
            state = self.ev_map[state]
            state_prime = self.ev_map[state_prime]
            state_action = self.apply_bf(state, action)
            state_action_prime = self.apply_bf(state_prime, self.greedy_policy(state_prime))
            x = state_action - self.gamma * state_action_prime
            self.matrix_A += np.matmul(state_action.reshape([state_action.shape[0], 1]), x.reshape([1, x.shape[0]]))
            self.vector_b += state_action * reward

    def fit(self):
        self.find_a_and_b()
        policy = np.matmul(np.linalg.inv(self.matrix_A), self.vector_b)
        return policy.reshape([self.num_actions, self.len_evs])


class LspiAgent:
    def __init__(self, eigenvectors, actions, max_samples=10 ** 6, source_of_samples=[]):
        self.eigenvectors = eigenvectors  # eigenvectors are a dict mapping ev number to ev
        self.source_of_samples = source_of_samples
        self.max_samples = max_samples
        self.len_evs = len(list(eigenvectors.values())[0])
        self.num_actions = len(actions)  # TODO: actions.n
        self.policy = np.zeros([self.num_actions, self.len_evs])
        self.epsilon = 1  # TODO
        self.action_space = actions
        self.model = Lstdq

    @staticmethod
    def random_tiebreak_argmax(x):
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            q_values = np.matmul(self.policy, state)
            action = self.random_tiebreak_argmax(q_values)
        return action

    def learn(self, stopping_criteria, max_out=100000):
        diff = stopping_criteria + 1
        i = 1
        for i in range(max_out):
            if diff > stopping_criteria:
                w_old = self.policy
                agent = self.model(self.num_actions, w_old, self.eigenvectors, self.source_of_samples)
                w_new = agent.fit()
                self.policy = w_new
                diff = np.linalg.norm(w_old - w_new)
                i += 1
                # print(w_new)
            else:
                break
        return w_new

    # def collect_experience(self, episodes, env, max_episode_length):
    #     new_samples = []
    #     for i in range(episodes):
    #         env.reset()
    #         state = env.state_features
    #         for _ in range(max_episode_length):
    #             action = self.epsilon_greedy(state)
    #             _, reward, done, info = env.step(action)
    #             state_prime = info["state_features"]
    #             new_samples.append([state, action, reward, state_prime])
    #             state = state_prime
    #             if done or len(new_samples) >= self.max_samples:
    #                 break
    #     self.source_of_samples += new_samples
    #     if len(self.source_of_samples) >= self.max_samples:  # if too many samples, only keep last N
    #         self.source_of_samples = self.source_of_samples[-self.max_samples:]

    def run(self, experience, stopping_criteria):
        # self.collect_experience(episodes=episodes, env=env, max_episode_length=max_episode_length)
        self.learn(stopping_criteria=stopping_criteria, source_of_samples=experience)


#############
### test ###
#############

samples = [[0, 0, 1, 1],
           [0, 1, 0, 2],
           [0, 2, 0, 3],
           [0, 3, 1, 2],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [1, 2, 1, 2],
           [1, 3, 0, 3],
           [2, 0, 0, 3],
           [2, 1, 1, 2],
           [2, 2, 0, 3],
           [2, 3, 1, 0],
           [3, 0, 0, 0],
           [3, 1, 1, 0],
           [3, 2, 1, 2],
           [3, 3, 0, 1]]

evs = {0: [1, 0, 0, 0],
       1: [0, 1, 0, 0],
       2: [0, 0, 1, 0],
       3: [0, 0, 0, 1]}

# x = Lstdq(num_actions=4,
#           policy=[[4, 2, 6, 2],
#                   [6, 0, 7, 2],
#                   [4, 3, 6, 7],
#                   [8, 3, 1, 2]],
#           eigenvectors=evs,
#           source_of_samples=samples)

y = LspiAgent(eigenvectors=evs, actions=[0, 1, 2, 3], max_samples=10 ** 6, source_of_samples=samples)
y.run(200, "x", .1)


#  next thing to do - make it work with "collect experience"
