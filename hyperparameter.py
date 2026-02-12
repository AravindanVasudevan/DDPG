import torch

print_episode = 100
render_episode = 500
n_training_episodes = 2000
buffer_size = 10000
batch_size = 256
tau = 0.005
gamma = 0.99
lr_a = 5e-3
lr_c = 5e-3
env_id = 'PandaReachDense-v3'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
agent_name = 'PandaReach'