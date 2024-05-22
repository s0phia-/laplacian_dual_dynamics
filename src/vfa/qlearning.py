from collections import defaultdict
import numpy as np
import random
import csv
import pandas as pd


class QLearning:
    def __init__(self,
                 n_actions: int = 0,
                 seed: float = 0):
        self.n_actions = n_actions
        self.alpha = 1
        self.gamma = .9
        self.epsilon = .15
        self.qq = defaultdict(lambda: np.zeros([n_actions]))
        np.random.seed = seed

    def reset(self):
        self.qq = defaultdict(lambda: np.zeros([self.n_actions]))

    def choose_action(self, state):
        """
        choose the action using epsilon greedy
        :param state: current state
        :param actions: use this if the after_states change per state. If the after_states are always the same it's cleaner to
        init after_states when making class
        :return: action selected by epsilon greedy
        """
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.n_actions - 1)
        else:
            action = self.argmax(self.qq[state])
        return action

    @staticmethod
    def argmax(x):
        """
        argmax that breaks ties randomly
        :param x: anything that can be turned into an NP array
        :return:
        """
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def learn(self, state, action, reward, state_):
        """
        implementation of a single Q learning update
        :param state:
        :param action:
        :param reward:
        :param state_: state prime
        """
        max_qq_s_ = np.max(self.qq[state_])
        qq_s_a = self.qq[state][action]
        update = self.alpha * (reward + self.gamma * max_qq_s_ - qq_s_a)
        self.qq[state][action] += update


class PlotAgent:
    def __init__(self, agent, env, results_path, steps=50000):
        self.agent = agent
        self.env = env
        self.results_path = results_path
        self.steps = steps

    def run_q(self):
        self.agent.reset()
        out = []
        step = 0
        state, _ = self.env.reset()
        state = tuple(state['agent'])
        total_return = 0
        episode = 0
        for s in range(self.steps):
            action = self.agent.choose_action(state)
            state_, reward, done, truncated, _ = self.env.step(action)
            state_ = tuple(state_['agent'])
            self.agent.learn(state, action, reward, state_)
            step += 1
            total_return += reward
            out.append([step, total_return, episode])
            state = state_
            if done:
                state, _ = self.env.reset()
                state = tuple(state['agent'])
                total_return = 0
                episode += 1
        return out

    def get_plotting_data(self, agents=50):
        def last(df):
            return df.tail(1)

        toplot = pd.DataFrame(data=[], columns=['steps', 'max_return', 'agent'])
        for a in range(agents):
            data = self.run_q()
            i = pd.DataFrame(data=data, columns=['steps', 'return', 'episode'])
            i['s'] = i.groupby(['episode'])['return'].transform(last)
            i['max_return'] = i.groupby(['episode'])['s'].transform('max')
            i['agent'] = a
            i = i[['steps', 'max_return', 'agent']]
            toplot = toplot._append(i)
        toplot = toplot.groupby(['steps'])['max_return'].mean().reset_index()
        toplot['datapoints'] = toplot['steps']
        toplot = toplot.rename(columns={'max_return': 'avg_return'})
        toplot.to_csv(self.results_path, sep=',', index=False)
