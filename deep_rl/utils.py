import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
def compute_G(rewards, gamma):
    """
    Args:
     - rewards: rewards of trajectory
    Out:
     - disc_rew: disc_rew[t] = G_t is the reward to go at time t
    ----
    Example:
    if rewards = [1, 2, 3], disc_rew = [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3]
    """
    T = len(rewards)
    disc_rew = np.zeros((T,))
    for t in range(T):
        for i in range(t,T):
            disc_rew[t] += rewards[i] * gamma**(i-t)
    return disc_rew




class Algorithm:

    def __init__(self,
                 env_id,
                 gamma,
                 folder=None,
                ):
        if folder is None:
            folder = "./results/other"
        self.env_id = env_id
        self.gamma = gamma
        self.env = gym.make(self.env_id)
        self.monitor_env = Monitor(self.env,
                                    folder,
                                    force=True,
                                    video_callable=lambda episode: True)




    def train(self, n_trajectories, n_episodes):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_episodes : int
            The number of gradient updates

        """

        rewards = []
        for episode in range(n_episodes):
            rewards.append(self.run_episode(n_trajectories))
            print(f'Episode {episode + 1}/{n_episodes}: rewards {round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}')

        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
        plt.show()

    def evaluate(self, render=False):
        """Evaluate the agent on a single trajectory
        """
        with torch.no_grad():
            env = self.monitor_env if render else self.env
            observation = env.reset()
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode = 0
            done = False

            while not done:
                action = self.policy(observation)
                observation, reward, done, info = env.step(int(action))
                observation = torch.tensor(observation, dtype=torch.float)
                reward_episode += reward

            env.close()
            if render:
                print(f'Reward: {reward_episode}')
            return reward_episode
