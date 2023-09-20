import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import device

class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)

        # self.mean_fc = nn.Linear(300, action_size)
        # self.std_fc = nn.Linear(300, action_size)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # mean = torch.tanh(self.mean_fc(x))
        # std = F.softplus(self.std_fc(x))
        # return mean, std
        return x
      
class Critic(nn.Module):

  def __init__(self, state_size, action_size):
      super(Critic, self).__init__()
      self.fc1 = nn.Linear(state_size, 400)
      self.fc2 = nn.Linear(400 + action_size, 300)
      self.fc3 = nn.Linear(300, 1)
    #   self.fc1 = nn.Linear(state_size + action_size, 128)
    #   self.fc2 = nn.Linear(128, 128)
    #   self.fc3 = nn.Linear(128, 32)
    #   self.fc4 = nn.Linear(32, 1)

  def forward(self, state, action):
      state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      action = torch.from_numpy(action).float().unsqueeze(0).to(device)
      value = F.relu(self.fc1(state))
      value = F.relu(self.fc2(torch.cat([value, action], dim = 2)))
      value = self.fc3(value)
    #   value = F.relu(self.fc1(torch.cat([state, action], 2)))
    #   value = F.relu(self.fc2(value))
    #   value = F.relu(self.fc3(value))
    #   value = self.fc4(value)
      return value