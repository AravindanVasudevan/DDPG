import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, state_size, action_size, h_dim1 = 256, h_dim2 = 128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))

        return action
      
class Critic(nn.Module):
    def __init__(self, state_size, action_size, h_dim1 = 256, h_dim2 = 128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim = -1)))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value