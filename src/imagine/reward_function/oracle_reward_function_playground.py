import numpy as np
from src.playground_env.reward_function import get_reward_from_state


class OracleRewardFunction:
    """
    Oracle Reward Function. Used to get ground truth rewards to evaluate success rate
    """
    def __init__(self, goal_sampler, params, convert_from_discovery_ids=False):
        self.descriptions = params['train_descriptions']
        self.obs_dim = params['dims']['obs'] // 2
        self.goal_sampler = goal_sampler
        self.convert_from_discovery_ids = convert_from_discovery_ids
        self.get_reward_from_state = get_reward_from_state
        self.env_params = params['env_params']

    def store(self, data):
        pass

    def update(self, epoch):
        pass

    def share_reward_function_to_all_cpus(self):
        pass

    def predict(self, state, goal_ids):
        goal_ids = np.atleast_1d(goal_ids.squeeze().astype(np.int))
        assert state.shape[0] == goal_ids.shape[0]

        rewards = - np.ones([state.shape[0]])
        for i in range(state.shape[0]):
            if self.convert_from_discovery_ids:
                g_id = -1 if goal_ids[i] == -1 else self.goal_sampler.id2oracleid[goal_ids[i]]
            else:
                g_id = goal_ids[i]

            if g_id < 0:
                rewards[i] = -1
            else:
                rewards[i] = (int(self.get_reward_from_state(state[i], self.descriptions[g_id])) == 1) - 1
        return rewards, None


    def eval_all_goals_from_episode(self, episode):
        return self.eval_all_goals_from_transition(episode, -1)


    def eval_all_goals_from_initial_state(self, episode):
        return self.eval_all_goals_from_transition(episode, 0)

    def eval_all_goals_from_transition(self, episode, ind_transition):
        state = episode['obs'][ind_transition]
        successes = []
        for i in self.descriptions:
            successes.append(int(self.get_reward_from_state(state, i, self.env_params)))
        return np.array(successes)


    def eval_goal_from_episode(self, episode, goal_id):
        state = episode['obs'][-1]
        return int(self.get_reward_from_state(state, self.descriptions[goal_id], self.env_params))
