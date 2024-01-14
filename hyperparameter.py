import torch

print_episode = 100
render_episode = 1000
n_training_episodes = 10000
buffer_size = 10000
batch_size = 100
tau = 0.05
gamma = 0.95
lr_a = 0.0005
lr_c = 0.0005
env_id = 'PandaReachDense-v3'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
agent_name = 'PandaReach'