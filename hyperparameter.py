import torch

print_episode = 1
render_episode = 20
n_training_episodes = 100
buffer_size = 1000000
batch_size = 64
tau = 0.005
max_t = 1000
max_t_sim = 100
gamma = 0.99
lr_a = 0.0001
lr_c = 0.0005
env_id = 'HalfCheetah-v4'
# env_id = 'Ant-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
agent_name = 'HalfCheetah'
# agent_name = 'Ant'