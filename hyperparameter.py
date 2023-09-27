import torch

print_episode = 20
render_episode = 100
n_training_episodes = 500
buffer_size = 1000000
batch_size  = 1024
tau = 0.001
max_t = 1000
max_t_sim = 100
gamma = 0.99
lr_a = 0.0001
lr_c = 0.0002
# env_id = 'HalfCheetah-v4'
env_id = 'Ant-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# agent_name = 'HalfCheetah'
agent_name = 'Ant'