import gym
import numpy as np
import pygame
from pygame.locals import *
import time
import sys

sys.path.append('../')
ENV_NAME = 'PlaygroundNavigationHuman-v1'

from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state
from src.playground_env.descriptions import generate_all_descriptions
from src.playground_env.env_params import get_env_params
"""
Playing script. Control the agent with the arrows, close the gripper with the space bar.
"""

env = gym.make(ENV_NAME, reward_screen=False, viz_data_collection=True)
pygame.init()

env_params = get_env_params()
train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)
all_descriptions = train_descriptions +  test_descriptions

# Select the goal to generate the scene.
goal_str = np.random.choice(all_descriptions)

env.reset()
env.unwrapped.reset_with_goal(goal_str)

while True:
    # init_render

    action = np.zeros([3])
    for event in pygame.event.get():
        if hasattr(event, 'key'):
            # J1
            if (event.key == K_DOWN):
                action[1] = -1
            elif event.key == K_UP:
                action[1] = 1
            # J2
            elif (event.key == K_LEFT):
                action[0] = -1
            elif event.key == K_RIGHT:
                action[0] = 1
            # J3
            elif event.key == K_SPACE:
                action[2] = 1

            elif event.key == K_q:
                stop = True
            if action.sum() != 0:
                time.sleep(0.05)
                break

    out = env.step(action)
    env.render()

    # Sample descriptions of the current state
    train_descr, test_descr, extra_descr = sample_descriptions_from_state(out[0], env.unwrapped.params)
    descr = train_descr + test_descr
    print(descr)

    # assert that the reward function works, should give positive rewards for descriptions sampled, negative for others.
    for d in descr:
        assert get_reward_from_state(out[0], d, env_params)
    for d in np.random.choice(list(set(all_descriptions) - set(descr)), size=20):
        assert not get_reward_from_state(out[0], d, env_params)


