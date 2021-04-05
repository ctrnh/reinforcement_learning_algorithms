#from pathlib import Path
#import base64
from rl_algorithms.deep_rl.networks import Model
from rl_algorithms.deep_rl import utils
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
#from pprint import pprint
#from pyvirtualdisplay import Display
#from IPython import display as ipythondisplay
#from IPython.display import clear_output
#import math


class ReinforceNetwork1(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()

        self.n_actions = n_actions
        self.state_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(in_features=self.state_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action



class ReinforceNetwork(Model):
    def __init__(self, state_dim, n_actions):
        super().__init__(input_size=state_dim,
                         hidden_size1=16,
                         hidden_size2=8,
                         output_size=n_actions)

        self.n_actions = n_actions
        self.state_dim = state_dim

    def forward(self, state):
        out = super().forward(state)
        return F.softmax(out,dim=0)





class Reinforce(utils.Algorithm):
    def __init__(self,
                env_id,
                gamma,
                learning_rate,
                folder):

        super().__init__(env_id=env_id, gamma=gamma)
        self.model = ReinforceNetwork(self.env.observation_space.shape[0], self.env.action_space.n)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


    def _compute_G2(self, rewards):
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
                disc_rew[t] += rewards[i] * self.gamma**(i-t)
        return disc_rew

    def _compute_G(self, rewards):
        return utils.compute_G(rewards=rewards, gamma=self.gamma)

    def policy(self, state):
        action = torch.multinomial(self.model.forward(state), 1)
        return action

    def run_episode(self, n_trajectories):
        reward_trajectories = []
        loss = 0

        for cur_traj in range(n_trajectories):
            done = False
            state = torch.tensor(self.env.reset(), dtype=torch.float)

            cur_traj_rewards = []
            cur_traj_pi = [] # cur_traj_pi[t] = pi(a_t|s_t)
            while not done:
                action = self.policy(state).item()
                cur_traj_pi.append(self.model.forward(state)[action])
                state, cur_reward, done, info = self.env.step(action)
                state = torch.tensor(state, dtype=torch.float)
                cur_traj_rewards.append(cur_reward)

            # compute q
            cur_traj_G = self._compute_G(cur_traj_rewards)
            reward_trajectories.append(cur_traj_G[0])
            for t in range(len(cur_traj_G)):
                loss += torch.log(cur_traj_pi[t]) * cur_traj_G[t]

        loss = - loss/n_trajectories


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.array(reward_trajectories)


env_id = 'CartPole-v1'
#env_id = 'Acrobot-v1'
#env_id = 'MountainCar-v0'
learning_rate = 0.01
gamma = 1


agent = Reinforce(env_id=env_id, learning_rate=learning_rate, gamma=gamma,folder="./results/reinforce")
agent.train(n_trajectories=50, n_episodes=50)


agent.evaluate(render=True)
