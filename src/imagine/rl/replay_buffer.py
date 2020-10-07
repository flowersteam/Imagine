import threading
from collections import deque
import numpy as np
import time

from mpi4py import MPI


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, goal_sampler, reward_function):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.bias_buffer = True
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        self.goal_sampler = goal_sampler
        self.reward_function = reward_function

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        self.buffers['g_str'] = np.array([None] * self.size)

        self.goals_indices = []
        self.imagined_goal_indices = []

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.pointer = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size, epoch):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['obs_2'] = buffers['obs'][:, 1:, :]
        init = time.time()
        out = self.sample_transitions(buffers,
                                      self.goals_indices,
                                      batch_size,
                                      epoch)
        transitions, replay_ratio_positive_rewards, replay_proba, replay_ratio_positive_per_goal, time_dict = out
        for key in (['r', 'obs_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key
        time_dict['time_buffer_sample'] = time.time() - init
        return transitions, replay_ratio_positive_rewards, replay_proba, replay_ratio_positive_per_goal, time_dict

    def add_imagined_goals_to_goals_reached_ids(self, discovered_goal_ids, imagined_inds, episode_batch, goals_reached_ids):

        if len(imagined_inds) > 0:  # when it is here, the reward function is also used to check for satisfied imagined goals and fill corresponding buffers
            final_obs = np.array([ep['obs'][-1] for ep in episode_batch])
            imagined_goals = np.array(discovered_goal_ids)[imagined_inds]
            # test 50 goals for each episode
            n_attempts = min(50, len(imagined_goals))
            goals_to_try = np.random.choice(imagined_goals, size=n_attempts, replace=False)
            obs = np.repeat(final_obs, n_attempts, axis=0)
            goals = np.tile(goals_to_try, final_obs.shape[0])
            rewards = self.reward_function.predict(state=obs, goal_ids=goals)[0]

            for i in range(len(episode_batch)):
                pos_goals = goals_to_try[np.where(rewards[i * n_attempts: (i + 1) * n_attempts] == 0)].tolist()
                goals_reached_ids[i] += pos_goals
        return goals_reached_ids

    def store_episode(self, episode_batch, goals_reached_ids):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """

        # update the set of discovered goal ids
        discovered_goal_ids = self.goal_sampler.feedback_memory['memory_id']
        for _ in range(len(discovered_goal_ids) - len(self.goals_indices)):
            self.goals_indices.append(deque())
        imagined_inds = np.argwhere(np.array(self.goal_sampler.feedback_memory['imagined']) == 1).flatten()
        goals_reached_ids = self.add_imagined_goals_to_goals_reached_ids(discovered_goal_ids, imagined_inds, episode_batch, goals_reached_ids)

        batch_size = len(episode_batch)
        assert batch_size == len(goals_reached_ids)

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # Maintain buffers for each goal reached to bias buffer sampling
            # An episode is added to a particular goal buffer if
            # the final transition satisfies that goal.
            if self.bias_buffer:
                for i in range(batch_size):
                    # remove old indices
                    if self.current_size == self.size:
                        for goal_buffer_ids in self.goals_indices:
                            if len(goal_buffer_ids) > 0:
                                if idxs[i] == goal_buffer_ids[0]:
                                    goal_buffer_ids.popleft()
                    # append new goal indices
                    for reached_id in goals_reached_ids[i]:
                        assert reached_id in discovered_goal_ids
                        ind_list = discovered_goal_ids.index(reached_id)
                        self.goals_indices[ind_list].append(idxs[i])

            # load inputs into buffers
            for i in range(batch_size):
                for key in self.buffers.keys():
                    self.buffers[key][idxs[i]] = episode_batch[i][key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1  # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # fifo memory
        if self.pointer + inc <= self.size:
            idx = np.arange(self.pointer, self.pointer + inc)
            self.pointer = self.pointer + inc
        else:
            overflow = inc - (self.size - self.pointer)
            idx_a = np.arange(self.pointer, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.pointer = overflow

        # update replay size
        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx
