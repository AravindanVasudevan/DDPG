import numpy as np

import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from ddpg import(
    OU_Noise,
    DDPG
)
from hyperparameter import(
    print_episode,
    render_episode,
    n_training_episodes,
    buffer_size,
    batch_size,
    tau,
    gamma,
    lr_a,
    lr_c,
    env_id
)

if __name__ == '__main__':
    env = gym.make(env_id, render_mode = 'rgb_array', renderer = 'Tiny')
    s_env = gym.make(env_id, render_mode = 'rgb_array', renderer = 'Tiny')
    
    state_size = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0] + env.observation_space['achieved_goal'].shape[0]
    action_size = env.action_space.shape[0]

    ou_noise = OU_Noise(input_size = action_size)

    ddpg_m = DDPG(state_size, action_size, buffer_size, batch_size, lr_a, lr_c, tau, gamma)
    
    rewards = []
    episodes = []

    for e in range(1, n_training_episodes + 1):
        ou_noise.reset()
        state_dict, _ = env.reset()
        state = np.concatenate((state_dict['observation'], state_dict['achieved_goal'], state_dict['desired_goal']))
        r = 0
        while True:
            exploration_noise = ou_noise.sample()
            action = ddpg_m.act(state, exploration_noise)
            next_state_dict, reward, terminated, truncated, _ = env.step(action) 
            next_state = np.concatenate((next_state_dict['observation'], next_state_dict['achieved_goal'], next_state_dict['desired_goal']))
            ddpg_m.add_data(state, action, reward, next_state, terminated, truncated)
            ddpg_m.learn()
            r += reward

            done = terminated or truncated

            if r <= -250:
                    done = True

            if done:
                 break
            
            state = next_state

        if e % print_episode == 0:
            print(f'Episode {e} Reward: {r}')
        
        if e % render_episode == 0 or e == 1:
            ddpg_m.render(e, s_env)

        rewards.append(r)
        episodes.append(e)

    plt.plot(episodes, rewards, label = 'Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.title('Training Performance')
    plt.show()

    print('Done!')