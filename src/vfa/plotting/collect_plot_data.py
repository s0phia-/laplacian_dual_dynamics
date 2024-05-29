import numpy as np


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
