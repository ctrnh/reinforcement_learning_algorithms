# Value Agent
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

class ValueNetwork(Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size=input_size,
                        hidden_size1=hidden_size,
                        hidden_size2=hidden_size,
                        output_size=output_size)

    def predict(self, x):
        return self(x).detach().numpy()[0]


class ActorNetwork(Model):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__(input_size=input_size,
                        hidden_size1=hidden_size,
                        hidden_size2=hidden_size,
                        output_size=action_size)

    def forward(self, x):
        out = super().forward(x)
        return F.softmax(out, dim=-1)


class A2C(utils.Algorithm):
    def __init__(self,
                 env_id,
                 gamma,
                 actor_lr,
                 value_lr,
                 folder=None):
        super().__init__(env_id=env_id, gamma=gamma,folder=folder)

        self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.env.action_space.n)

        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(), lr=value_lr)
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=actor_lr)

    def _compute_returns(self, rewards, dones, next_value):
        """Returns the cumulative discounted rewards
        Args:
        - rewards : array of shape (batch_size,)
        - dones : array of bool of shape (batch_size,)
        - next_value :  value of the next state given by the value network

        Out:
        - returns: cumulative discounted rewards

        """
        rewards = list(rewards) + [next_value]
        dones = list(dones) + [1]
        idx_dones = [-1] + list(np.argwhere(dones).flatten())
        disc_rew = np.zeros((len(rewards),))

        for i in range(len(idx_dones) - 1):
            start_idx = idx_dones[i] + 1
            end_idx = idx_dones[i+1]
            T = end_idx - start_idx + 1
            for t in range(T):
                for i in range(t,T):
                    disc_rew[start_idx + t] += rewards[start_idx + i] * self.gamma**(i-t)

        return disc_rew[:-1]

    def train(self, epochs, batch_size):
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []

        for epoch in range(epochs):
            for i in range(batch_size):
                observations[i] = observation
                actions[i] = self.policy(torch.tensor(observation, dtype=torch.float))
                values[i] =  self.value_network.predict(torch.tensor(observation, dtype=torch.float))
                observation, rewards[i], dones[i], info = self.env.step(actions[i])
                if dones[i]:
                    observation = self.env.reset()

            # If our episode didn't end on the last step: estimate value for the last state
            if dones[-1]:
                next_value = 0
            else:
                done = False
                next_value = self.value_network.forward(torch.tensor(observation, dtype=torch.float))


            episode_count += sum(dones)
            returns = self._compute_returns(rewards, dones,  next_value)
            advantages = returns - values

            self.optimize_model(observations, actions, returns, advantages)


            if epoch % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')


                if rewards_test[-1].mean() > 490 and epoch != epochs -1:
                    print('Early stopping')
                    break
                observation = self.env.reset()


        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        plt.show()
        print(f'Number of episodes:{episode_count}')

    def optimize_model(self, observations, actions, returns, advantages):
       # actions = F.one_hot(torch.tensor(actions), self.env.action_space.n)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)

        # MSE for Value function
        V_theta = self.value_network.forward(observations)
        loss = F.mse_loss(V_theta, returns)
        self.value_network_optimizer.zero_grad()
        loss.backward()
        self.value_network_optimizer.step()

        # Actor
        actor_loss = 0
        for t in range(len(observations)):
            pi = self.actor_network.forward(observations[t])[actions[t]]
            actor_loss += torch.log(pi) * advantages[t]

        actor_loss = - actor_loss
        self.actor_network_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network_optimizer.step()

    def policy(self, state):
        action = torch.multinomial(self.actor_network(state), 1)
        return action



env_id = 'CartPole-v1'
#env_id = 'Acrobot-v1'
#env_id = 'MountainCar-v0'
value_learning_rate = 0.001
actor_learning_rate = 0.001
gamma = 0.99


agent = A2C(env_id=env_id,
                gamma=gamma,
                value_lr=value_learning_rate,
                actor_lr=actor_learning_rate,
                folder="./results/actor_critic"
                )
rewards = agent.train(epochs=1000, batch_size=256)

agent.evaluate(render=True)
