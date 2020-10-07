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
if 'Navigation' in ENV_NAME:
    from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state, \
        supply_on_furniture
    from src.playground_env.descriptions import train_descriptions, test_descriptions, extra_descriptions
env = gym.make(ENV_NAME, reward_screen=False, viz_data_collection=True)
pygame.init()

stop = False
n_dim = 10
# seed = np.random.randint(300)
# np.random.seed(seed)
# env.seed(seed)

save_to_png = True
output_dir = 'test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for ep in range(9):
    env.reset()

    if ep == 0:
        goal_str = 'Grasp red dog'
    else:
        goal_str = random.choice(env.known_goals_descr)
    env.unwrapped.reset_with_goal(goal_str)

    # init_render
    frame_count = 100 * ep
    env.step(np.zeros([n_dim]))
    env.render(goal_str)
    if save_to_png:
        for n in range(15):
            pygame.image.save(env.viewer, output_dir + "/frame_{}.png".format(str(frame_count).zfill(4)))
            frame_count += 1
        frame_count += 1

    while not stop:
        action = np.zeros([n_dim])
        for event in pygame.event.get():
            if hasattr(event, 'key'):
                # J1
                if (event.key == K_d):
                    action[0] = 1
                elif event.key == K_a:
                    action[0] = -1
                # J2
                elif (event.key == K_w):
                    action[1] = 1
                elif event.key == K_s:
                    action[1] = -1
                # J3
                elif event.key == K_SPACE:
                    action[2] = 1
                elif event.key == K_d:
                    action[2] = -1
                # HOLD
                elif event.key == K_r:
                    action[3] = 1
                elif event.key == K_f:
                    action[3] = -1
                # # Push button

                # Control Agent X
                elif event.key == K_u:
                    action[4] = 1  # usually 5
                elif event.key == K_j:
                    action[4] = -1
                # Control Agent Y
                elif event.key == K_i:
                    action[5] = 1  # usually  6
                elif event.key == K_k:
                    action[5] = -1

                elif event.key == K_DOWN:
                    stop = True
                if action.sum() != 0:
                    time.sleep(0.05)
                    break

        out = env.step(action)
        env.render(goal_str)
        if action.sum() != 0:
            ici = 0
            if save_to_png:
                pygame.image.save(env.viewer, output_dir + "/frame_{}.png".format(str(frame_count).zfill(4)))
                frame_count += 1
        if save_to_png:
            if frame_count > 40 + 100 * ep:
                train, test, extra = sample_descriptions_from_state(out[0])
                sp_descr = random.choice(train)
                for descr in train:
                    if 'Grasp' in descr:
                        sp_descr = descr
                for descr in train:
                    if 'Grow' in descr:
                        sp_descr = descr

                env.set_SP_feedback(sp_descr)

                for n in range(22):
                    env.render(goal_str)
                    pygame.image.save(env.viewer, output_dir + "/frame_{}.png".format(str(frame_count).zfill(4)))
                    if n > 10:
                        env.update_known_goal_position(n - 10)
                    frame_count += 1
                print(sp_descr)
                env.update_known_goals_list()
                break

stop = 1
