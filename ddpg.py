import copy
import random
import torch
import numpy as np
import torch.optim as optim
from collections import namedtuple, deque
from model import (
    Actor,
    Critic
)
from hyperparameter import device

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'terminated', 'truncated'])

    def add(self, state, action, reward, next_state, terminated, truncated):
        extend = self.experiences(state, action, reward, next_state, terminated, truncated)
        self.memory.append(extend)

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        terminates = torch.from_numpy(np.vstack([e.terminated for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        truncates = torch.from_numpy(np.vstack([e.truncated for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, terminates, truncates

    def __len__(self):
        return len(self.memory)
    
class OU_Noise:

    def __init__(self, action_size, mu = 0, theta = 0.15, sigma = 0.1):
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state
    
class DDPG:

    def __init__(self, state_size, action_size, buffer_size, batch_size, lr_a, lr_c, tau, gamma):
        self.actor = Actor(state_size, action_size).to(device)
        self.target_actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size).to(device)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr = lr_a)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr = lr_c)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # self.update_networks(tau = 1)
    
    def act(self, state, noise):
        # if noise == None:
        #     action = self.actor(state)
        #     action = action.squeeze().detach().cpu().numpy()
        # else:    
        #     action = self.actor(state)
        #     action = action.squeeze().detach().cpu().numpy()
        #     action = np.clip(action + noise, -1, 1)
        action = self.actor(state)
        action = action.squeeze().detach().cpu().numpy()
        action = np.clip(action + noise, -1, 1)
        return action
    
    def add_data(self, state, action, reward, next_state, terminated, truncated):
        self.memory.add(state, action, reward, next_state, terminated, truncated)
    
    def update_networks(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, terminates, truncates = self.memory.sample()

            states = states.cpu().numpy()
            next_states = next_states.cpu().numpy()
            actions = actions.cpu().numpy()

            with torch.no_grad():
                next_actions = self.target_actor(next_states)
                next_actions = next_actions.squeeze().detach().cpu().numpy()
                next_values = self.target_critic(next_states, next_actions)
                target_q_values = rewards + self.gamma * (1 - torch.logical_or(terminates, truncates).float()) * next_values

            current_q_values = self.critic(states, actions)
    
            critic_loss = (current_q_values - target_q_values).pow(2).mean()
            # critic_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

            actor_loss = -self.critic(states, self.actor(states).squeeze().detach().cpu().numpy()).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            self.update_networks(tau = self.tau)
            # print(f'Actor Loss: {actor_loss} | Critic Loss: {critic_loss}')