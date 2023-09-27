import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import device

class Actor(nn.Module):

    def __init__(self, state_size, action_size, h_dim1 = 400, h_dim2 = 300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, action_size)
        self.ln1 = nn.LayerNorm(h_dim1)
        self.ln2 = nn.LayerNorm(h_dim2)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))

        return x
      
class Critic(nn.Module):

  def __init__(self, state_size, action_size, h_dim1 = 400, h_dim2 = 300):
      super(Critic, self).__init__()
      self.fc1 = nn.Linear(state_size, h_dim1)
      self.fc2 = nn.Linear(h_dim1 + action_size, h_dim2)
      self.fc3 = nn.Linear(h_dim2, 1)
      self.ln1 = nn.LayerNorm(h_dim1)
      self.ln2 = nn.LayerNorm(h_dim2)

      nn.init.xavier_normal_(self.fc1.weight)
      nn.init.xavier_normal_(self.fc2.weight)
      nn.init.xavier_normal_(self.fc3.weight)

  def forward(self, state, action):
      state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      action = torch.from_numpy(action).float().unsqueeze(0).to(device)
      value = F.relu(self.ln1(self.fc1(state)))
      value = F.relu(self.ln2(self.fc2(torch.cat([value, action], dim = 2))))
      value = self.fc3(value)

      return value

# class Critic(nn.Module):

#   def __init__(self, state_size, action_size, h_dim1 = 400, h_dim2 = 300):
#       super(Critic, self).__init__()
#       self.fc1 = nn.Linear(state_size + action_size, h_dim1)
#       self.fc2 = nn.Linear(h_dim1, h_dim2)
#       self.fc3 = nn.Linear(h_dim2, 1)
#       self.ln1 = nn.LayerNorm(h_dim1)
#       self.ln2 = nn.LayerNorm(h_dim2)

        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

#   def forward(self, state, action):
#       state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#       action = torch.from_numpy(action).float().unsqueeze(0).to(device)
#       value = F.relu(self.ln1(self.fc1(torch.cat([state, action], dim = 2))))
#       value = F.relu(self.ln2(self.fc2(value)))
#       value = self.fc3(value)

#       return value