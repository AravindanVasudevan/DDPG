import torch

print_step = 20
render_step = 100
n_training_episodes = 1000
buffer_size = 100000
batch_size  = 256
tau = 0.001
max_t = 1000
max_t_sim = 200
gamma = 0.995
lr_a = 1e-4
lr_c = 1e-4
env_id = 'HalfCheetah-v4'
# env_id = 'Ant-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
agent_name = 'HalfCheetah'
# agent_name = 'Ant'