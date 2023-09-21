import torch

print_step = 20
render_step = 100
n_training_episodes = 500
buffer_size = 100000
batch_size  = 128
tau = 0.01
max_t = 4000
max_t_sim = 100
gamma = 0.99
lr_a = 0.0001
lr_c = 0.0001
# env_id = 'HalfCheetah-v4'
env_id = 'Ant-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# agent_name = 'HalfCheetah'
agent_name = 'Ant'