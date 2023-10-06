import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hyperparameter import device

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, h_dim1 = 512, h_dim2 = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x
      
class Critic(nn.Module):
    def __init__(self, state_size, action_size, h_dim1 = 512, h_dim2 = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1 + action_size, h_dim2)
        self.fc3 = nn.Linear(h_dim2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(device)
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(torch.cat([value, action], dim = 2)))
        value = self.fc3(value)
        
        return value