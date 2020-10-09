import gym
import numpy as np
import pygame
from pygame.locals import *
import time
import sys
import random
import os

sys.path.append('../..')

ENV_NAME = 'PlaygroundNavigationHuman-v1'

from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state
from src.playground_env.descriptions import generate_all_descriptions
from src.playground_env.env_params import get_env_params
env = gym.make(ENV_NAME, reward_screen=False, viz_data_collection=True)
pygame.init()

stop = False


save_to_png = True
output_dir = 'test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

env_params = get_env_params()
train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)
all_descriptions = train_descriptions +  test_descriptions

while True:
    goal_str = np.random.choice(train_descriptions)
    env.reset()
    env.unwrapped.reset_with_goal(goal_str)

    # init_render

    while not stop:
        action = np.zeros([3])
        for event in pygame.event.get():
            if hasattr(event, 'key'):
                # J1
                if (event.key == K_d):
                    action[0] = 1
                elif event.key == K_q:
                    action[0] = -1
                # J2
                elif (event.key == K_s):
                    action[1] = 1
                elif event.key == K_w:
                    action[1] = -1
                # J3
                elif event.key == K_SPACE:
                    action[2] = 1
                elif event.key == K_n:
                    action[2] = -1

                elif event.key == K_DOWN:
                    stop = True
                if action.sum() != 0:
                    time.sleep(0.05)
                    break

        out = env.step(action)
        env.render()
        # print(env.unwrapped.objects)
        train_descr, test_descr, extra_descr = sample_descriptions_from_state(out[0], env.unwrapped.params)
        # print(extra_descr)
        descr = train_descr + test_descr
        for d in descr:
            if not get_reward_from_state(out[0], d, env_params):
                stop = 1
        for d in np.random.choice(list(set(all_descriptions) - set(descr)), size=20):
            if get_reward_from_state(out[0], d, env_params):
                stop = 1



stop = 1
