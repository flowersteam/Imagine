import sys

sys.path.append('../../')
from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state
from src.playground_env.descriptions import train_descriptions, test_descriptions, extra_descriptions

import numpy as np
import os
import gym
import pickle
import time


class GRBFTrajectory(object):
    def __init__(self, n_dims, sigma, steps_per_basis, max_basis):
        self.n_dims = n_dims
        self.sigma = sigma
        self.alpha = - 1. / (2. * self.sigma ** 2.)
        self.steps_per_basis = steps_per_basis
        self.max_basis = max_basis
        self.precomputed_gaussian = np.zeros(2 * self.max_basis * self.steps_per_basis)
        for i in range(2 * self.max_basis * self.steps_per_basis):
            self.precomputed_gaussian[i] = self.gaussian(self.max_basis * self.steps_per_basis, i)

    def gaussian(self, center, t):
        return np.exp(self.alpha * (center - t) ** 2.)

    def trajectory(self, weights):
        n_basis = len(weights) // self.n_dims
        weights = np.reshape(weights, (n_basis, self.n_dims)).T
        steps = self.steps_per_basis * n_basis
        traj = np.zeros((steps, self.n_dims))
        for step in range(steps):
            g = self.precomputed_gaussian[
                self.max_basis * self.steps_per_basis + self.steps_per_basis - 1 - step::self.steps_per_basis][:n_basis]
            traj[step] = np.dot(weights, g)
        return np.clip(traj, -1., 1.)


if __name__ == "__main__":
    NB_EPS = 50
    DIM_ACT = 3
    NB_STEPS = 50

    dataset_name = 'dataset_playground'
    data_raw = '../data/raw'
    if not os.path.exists(data_raw):
        os.makedirs(data_raw)
    path = data_raw + '/{}.pk'.format(dataset_name)

    trajectory_generator = GRBFTrajectory(n_dims=DIM_ACT, sigma=3, steps_per_basis=6, max_basis=10)

    env = gym.make('PlaygroundNavigation-v1')
    id2description = dict()
    description2id = dict()
    id_description = 0

    all_descr = train_descriptions + test_descriptions
    all_descriptions_ids = []
    all_obs = []

    times = []
    t_i = time.time()
    for ep in range(NB_EPS):

        if not (ep + 1) % 500:
            times.append(time.time() - t_i)
            t_i = time.time()
            print((ep - 1) / NB_EPS, ', ETA: ', (NB_EPS - ep) / 500 * np.mean(times))
        m = np.clip(2. * np.random.random(10 * DIM_ACT) - 1., -1, 1)
        actions = trajectory_generator.trajectory(m)

        obs_episode = []
        descriptions_ids_episode = []

        init_obs = env.reset()
        d = all_descr[ep % len(all_descr)]
        obs_episode.append(env.unwrapped.reset_with_goal(d))
        train_descr, test_descr, extra_descr = sample_descriptions_from_state(obs_episode[-1], modes=[1, 2])
        sampled_descriptions = train_descr.copy() + test_descr.copy()

        descriptions_ids = []
        for description in sampled_descriptions:
            if description not in id2description.values():
                id2description[id_description] = description
                description2id[description] = id_description
                id_description += 1
            descriptions_ids.append(description2id[description])
        descriptions_ids_episode.append(descriptions_ids)

        for t in range(NB_STEPS):
            out = env.step(actions[t])
            obs_episode.append(out[0].copy())
            train_descr, test_descr, extra_descr = sample_descriptions_from_state(obs_episode[-1],
                                                                                  modes=[1, 2])
            sampled_descriptions = train_descr.copy() + test_descr.copy()

            descriptions_ids = []
            for description in sampled_descriptions:
                if description not in id2description.values():
                    id2description[id_description] = description
                    description2id[description] = id_description
                    id_description += 1
                descriptions_ids.append(description2id[description])
            descriptions_ids_episode.append(descriptions_ids)

        all_descriptions_ids.append(descriptions_ids_episode)
        all_obs.append(obs_episode)

    dataset = dict(obs=np.array(all_obs), descriptions_ids=all_descriptions_ids, id2description=id2description,
                   description2id=description2id)
    with open(path, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    stop = 1
