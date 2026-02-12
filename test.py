import gymnasium as gym
import numpy as np

import panda_gym

env = gym.make("PandaReachJointsDense-v3", render_mode="human")
observation, _ = env.reset()
print(observation)
for _ in range(1000):
    state_id = env.save_state()

    # Sample 5 actions and choose the one that yields the best reward.
    best_reward = -np.inf
    best_action = None
    for _ in range(5):
        env.restore_state(state_id)
        action = env.action_space.sample()
        observation, reward, _, _, _ = env.step(action)
        if reward > best_reward:
            best_reward = reward
            best_action = action

    env.restore_state(state_id)
    env.remove_state(state_id)  # discard the state, as it is no longer needed

    # Step with the best action
    observation, reward, terminated, truncated, info = env.step(best_action)
    print(f'best_action is {best_action}')
    if terminated:
        observation, info = env.reset()
x = np.concatenate((observation['observation'], observation['achieved_goal'], observation['desired_goal']))
print(x)
env.close()