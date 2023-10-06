import copy
import random
import torch
import numpy as np
import torch.nn.functional as F
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

    def __init__(self, action_size, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_size = action_size
        self.mu = mu * np.ones_like(action_size)
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
    
    def sample_zero(self):
        return np.zeros_like(self.mu)
    
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

    def act(self, state, noise):
        self.actor.eval()
        action = self.actor(state)
        action = action.squeeze().detach().cpu().numpy()
        action = np.clip(action + noise, -1, 1)
        self.actor.train()
        return action
    
    def add_data(self, state, action, reward, next_state, terminated, truncated):
        self.memory.add(state, action, reward, next_state, terminated, truncated)
    
    def update_networks(self, tau):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, terminates, truncates = self.memory.sample()
            states = states.cpu().numpy()
            next_states = next_states.cpu().numpy()
            actions = actions.cpu().numpy()

            next_actions = self.target_actor(next_states)
            next_actions = next_actions.squeeze().detach().cpu().numpy()
            next_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + self.gamma * (1 - torch.logical_or(terminates, truncates).float()) * next_values

            current_q_values = self.critic(states, actions)
    
            self.opt_critic.zero_grad()
            # critic_loss = (current_q_values - target_q_values).pow(2).mean()
            critic_loss = F.mse_loss(current_q_values, target_q_values)
            # critic_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            critic_loss.backward()
            self.opt_critic.step()

            self.opt_actor.zero_grad()
            self.actor.train()
            actor_loss = -1 * self.critic(states, self.actor(states).squeeze().detach().cpu().numpy()).mean()
            actor_loss.backward()
            self.opt_actor.step()
            self.actor.eval()
            self.critic.eval()

            self.update_networks(tau = self.tau)