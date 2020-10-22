import numpy as np
from mpi4py import MPI
import random

from src.playground_env.reward_function import sample_descriptions_from_state


class SocialPartner:
    def __init__(self, oracle_reward_function, goal_sampler, feedback_strategy='exhaustive', p_availability=1.,
                 params=None):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.oracle_reward_function = oracle_reward_function
        self.strategy = feedback_strategy
        self.goal_sampler = goal_sampler
        self.p_availability = p_availability
        self.count = 0
        self.feedback_fun = self.get_exhaustive_feedback_playground
        self.params = params

    def is_available(self):
        # always available the n first times
        if self.rank == 0:
            if self.count < 10:
                self.count += 1
                available = True
            else:
                available = np.random.rand() < self.p_availability
        else:
            available = None
        available = MPI.COMM_WORLD.bcast(available, root=0)
        return available

    def get_feedback(self, episodes):
        all_goals_reached_str = []
        all_test_descr = []
        all_extra_descr = []
        all_train_descr = []
        for ep in episodes:
            goals_reached_str, test_descr, extra_descr = self.feedback_fun(ep)
            all_goals_reached_str.append(goals_reached_str)
            all_train_descr += goals_reached_str.copy()
            all_test_descr += test_descr.copy()
            all_extra_descr += extra_descr.copy()

        return all_goals_reached_str, all_train_descr.copy(), all_test_descr.copy(), all_extra_descr.copy()

    def get_one_pos_one_neg_feedback(self, episodes):
        all_goals_reached_str, _, _, _ = self.get_feedback(episodes)
        pos = [random.choice(all_goals_reached_str[i]) for i in range(len(episodes))]
        all_neg = [set(self.goal_sampler.feedback_memory['string']) - set(all_goals_reached_str[i]) for i in range(len(episodes))]
        neg = [random.choice(list(all_neg[i])) for i in range(len(episodes))]
        return pos, neg

    def get_exhaustive_feedback_playground(self, episode):
        train_descr, test_descr, extra_descr = sample_descriptions_from_state(episode['obs'][-1],
                                                                              self.params['env_params'])
        return train_descr, test_descr, extra_descr
