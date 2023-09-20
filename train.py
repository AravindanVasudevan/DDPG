import gym
import imageio
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ddpg import(
    ReplayBuffer,
    OU_Noise,
    DDPG
)
from hyperparameter import(
    print_step,
    render_step,
    n_training_episodes,
    update_step,
    start_size,
    buffer_size,
    batch_size,
    tau,
    max_t,
    max_t_sim,
    gamma,
    lr_a,
    lr_c,
    env_id,
    agent_name
)


def render(actor, max_t_sim, s_environment, episode):
    actor.eval()
    frames = []
    sim_reward = 0
    state, _ = s_environment.reset()
    with torch.no_grad():
        for _ in range(max_t_sim):
            action = actor(state)
            action = action.squeeze().detach().cpu().numpy()

            exploration_noise = ou_noise.sample()
            action = action + exploration_noise
            action = np.clip(action, -1, 1)

            next_state, reward, terminated, truncated, _ = s_environment.step(action)  
            sim_reward += reward
            frame = s_environment.render()
            frames.append(frame)
            imageio.mimsave(f'simulations/simulation_episode_{episode}.gif', frames)

            if terminated or truncated:
                break

            state = next_state

    print(f'simulation for training episode {episode} is saved and the reward obtained is {sim_reward}')
    actor.train()

if __name__ == '__main__':
    env = gym.make(env_id)
    s_env = gym.make(env_id, render_mode = 'rgb_array')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    ddpg_m = DDPG(state_size, action_size, buffer_size, batch_size, start_size, lr_a, lr_c, tau, gamma)

    rewards = []
    episodes = []
    best_reward = 0

    for e in range(1, n_training_episodes + 1):
        ou_noise = OU_Noise(action_size = action_size)

        state, _ = env.reset()
        r = 0
        for step in range(max_t):
            exploration_noise = ou_noise.sample()
            action = ddpg_m.act(state, exploration_noise)

            next_state, reward, terminated, truncated, _ = env.step(action)
            ddpg_m.add_data(state, action, reward, next_state, terminated, truncated)

            if step % update_step == 0:
                ddpg_m.learn()

            r += reward

            if terminated or truncated:
                break
        
        if e % render_step == 0:
            render(ddpg_m.actor, max_t_sim, s_env, e)
        
        if e % print_step == 0:
            print(f'Episode {e} Reward: {r}')

        if r > best_reward or e == 1:
            best_reward = r
            torch.save({'model_state_dict': ddpg_m.actor.state_dict()}, f'checkpoints/{agent_name}_best_actor_checkpoint.pth')
            torch.save({'model_state_dict': ddpg_m.critic.state_dict()}, f'checkpoints/{agent_name}_best_critic_checkpoint.pth')
            print(f'Saving the best model checkpoint with the reward {r} obtained at episode {e}')
        
        rewards.append(r)
        episodes.append(e)

    torch.save({'model_state_dict': ddpg_m.actor.state_dict()}, f'checkpoints/{agent_name}_actor_checkpoint.pth')
    torch.save({'model_state_dict': ddpg_m.critic.state_dict()}, f'checkpoints/{agent_name}_critic_checkpoint.pth')

    plt.plot(episodes, rewards)
    plt.show()

    print('Done!')