import time
from collections import deque

import numpy as np
from mpi4py import MPI
from src.utils.util import mpi_average


class ExplorationTracker:
    def __init__(self, params):
        self.params = params
        self.test_descr = sorted(params['test_descriptions'])
        self.train_descr = sorted(params['train_descriptions'])
        self.extra_descr = sorted(params['extra_descriptions'])
        self.all_descriptions = sorted(list(set(self.train_descr + self.test_descr + self.extra_descr)))
        self.metrics = dict(counter_since_begininng=dict())
        for descr in self.train_descr + self.test_descr + self.extra_descr:
            self.metrics['counter_since_begininng'][descr] = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics['rewards_last_state']=dict()
        self.metrics['reward_exploration_score_train'] = None
        self.metrics['count_reward_test_set'] = None
        self.metrics['count_reward_train_set'] = None
        self.metrics['count_reward_extra_set'] = None
        for descr in self.train_descr + self.test_descr + self.extra_descr:
            self.metrics['rewards_last_state'][descr] = 0

    def get_average_rarity(self):
        return np.mean([(1 / (self.metrics['counter_since_begininng'][descr] + 1)) for descr in self.all_descriptions])

    def update(self, episode_list, train_descr, test_descr, extra_descr):
        # track number of reward and exploration score
        explo_score = 0
        descriptions = sorted(list(set(train_descr + test_descr + extra_descr)))
        prev_counters = self.metrics['counter_since_begininng'].copy()
        for descr in descriptions:
            self.metrics['rewards_last_state'][descr] += 1
            self.metrics['counter_since_begininng'][descr] += 1
            explo_score += (1 / (prev_counters[descr] + 1))
        explo_score /= self.get_average_rarity()

        self.metrics['reward_exploration_score_train'] = explo_score

        # count rewards per set
        self.metrics['count_reward_train_set'] = np.sum([self.metrics['rewards_last_state'][descr] for descr in self.train_descr] )
        self.metrics['count_reward_test_set'] = np.sum([self.metrics['rewards_last_state'][descr] for descr in self.test_descr] )
        self.metrics['count_reward_extra_set'] = np.sum([self.metrics['rewards_last_state'][descr] for descr in self.extra_descr] )



class DataProcessor:
    def __init__(self, reward_function, oracle_reward_function, goal_sampler, params):

        # from the data generated through environment interactions and social partner's feedbacks
        # This object infers positive and negative samples and create examples to learn the reward function,
        # updates statistics and metrics about the reward function,
        # updates statistics and metrics about goals,
        # can save a dataset of episodes to learn the reward function offline

        self.reward_function = reward_function
        self.oracle_reward_function = oracle_reward_function
        self.goal_sampler = goal_sampler
        self.params = params
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.rollout_batch_size = params['experiment_params']['rollout_batch_size']
        self.n_cpus = params['experiment_params']['n_cpus']

        # track metrics for reward function
        self.stats_confusion_rew_func = [[deque(maxlen=100), deque(maxlen=100)] for _ in
                                         range(len(params['train_descriptions']))]

        # Track return histories
        self.training_return_histories = [deque(maxlen=100) for _ in range(len(params['train_descriptions']))]
        self.evaluation_return_histories = [deque(maxlen=100) for _ in range(len(params['train_descriptions']))]

        if params['experiment_params']['save_obs']:
            self.states_to_save = deque()

        if self.rank == 0:
            self.exploration_tracker = ExplorationTracker(self.params)

    def clear_memory_states_to_save(self):
        self.states_to_save = deque()

    def process(self, current_episode, epoch, episodes, partner_available,
                feedbacks_str, train_descr, test_descr, extra_descr):

        rank = MPI.COMM_WORLD.Get_rank()
        time_dict = dict()
        # logger.info("D1: " + str(rank))

        # Gather all feedbacks from the different cpus
        all_feedbacks_str = MPI.COMM_WORLD.gather(feedbacks_str, root=0)
        all_train_descr  = MPI.COMM_WORLD.gather(train_descr, root=0)
        all_test_descr  = MPI.COMM_WORLD.gather(test_descr, root=0)
        all_extra_descr  = MPI.COMM_WORLD.gather(extra_descr, root=0)

        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            # Put everything in the same list
            all_episodes_list = []
            all_train_descr_list = []
            all_test_descr_list = []
            all_extra_descr_list = []
            for i in range(len(all_episodes)):
                all_episodes_list += all_episodes[i]
                all_train_descr_list += all_train_descr[i]
                all_test_descr_list += all_test_descr[i]
                all_extra_descr_list += all_extra_descr[i]

            self.exploration_tracker.update(all_episodes_list, train_descr, test_descr, extra_descr)

        if partner_available:

            if self.rank == 0:

                all_feedbacks_str_list = []
                for i in range(len(all_episodes)):
                    all_feedbacks_str_list += all_feedbacks_str[i]

                # # # # # #
                # Run inference to complete social partner's feedbacks
                # # # # # #

                timee = time.time()
                out = self.infer_from_social_partner_feedbacks(all_feedbacks_str_list)
                goals_reached_str, goals_not_reached_str, new_goals_str = out
                time_dict.update(time_infer_social_partner=time.time() - timee)

                # # # # # #
                # Update the goal sampler
                # # # # # #
                timee = time.time()
                # Update list of discovered goals
                self.goal_sampler.update_discovered_goals(new_goals_str, current_episode, epoch)
                # Update goal statistics (for intrinsic motivations)
                goals_reached_ids, goals_not_reached_ids = self.goal_sampler.update(current_episode=current_episode,
                                                                                    all_episodes=all_episodes_list,
                                                                                    partner_available=partner_available,
                                                                                    goals_reached_str=goals_reached_str,
                                                                                    goals_not_reached_str=goals_not_reached_str)
                time_dict.update(time_goal_sampler=time.time() - timee)

                # # # # # #
                # Create training samples for learning the reward function
                # # # # # #
                timee = time.time()
                reward_data = self.convert_feedbacks_to_reward_samples(all_episodes_list,
                                                                       goals_reached_ids,
                                                                       goals_not_reached_ids)

                # Update reward metrics
                self.update_reward_metrics(all_episodes_list)
                time_dict.update(time_get_classif_samples=time.time() - timee)
            else:
                reward_data = None
        else:
            if self.rank == 0:
                goals_reached_ids, _ = self.goal_sampler.update(current_episode=current_episode,
                                                                all_episodes=all_episodes_list,
                                                                partner_available=partner_available,
                                                                goals_reached_str=[],
                                                                goals_not_reached_str=[])

            reward_data = None

        # Save episodes as dataset
        if self.params['experiment_params']['save_obs'] and self.rank == 0:
            self.states_to_save += all_episodes_list

        # # # # # #
        # Scatter data
        # # # # # #

        timee = time.time()
        if self.rank == 0:
            to_scatter_reached_ids = []
            for i in range(self.n_cpus):
                tmp_reached_ids = []
                for j in range(self.rollout_batch_size):
                    tmp_reached_ids.append(goals_reached_ids[i * self.rollout_batch_size + j])
                to_scatter_reached_ids.append(tmp_reached_ids)
        else:
            to_scatter_reached_ids = None
        goals_reached_ids = MPI.COMM_WORLD.scatter(to_scatter_reached_ids, root=0)

        # share goal sampler info to all cpus
        self.goal_sampler.share_info_to_all_cpus()
        time_dict.update(time_scatter_data=time.time() - timee)
        return reward_data, goals_reached_ids, time_dict

    def update_reward_metrics(self, all_episodes_list):
        if self.params['conditions']['reward_function'] != 'oracle':
            for ep in all_episodes_list:
                if np.random.rand() < 0.1:
                    oracle_inds = []
                    for el in np.array(self.goal_sampler.feedback_memory['oracle_id']):
                        if el is not None:
                            oracle_inds.append(el)
                    oracle_inds = np.array(oracle_inds)
                    oracle_successes = self.oracle_reward_function.eval_all_goals_from_episode(ep)[oracle_inds]
                    learned_successes = self.reward_function.eval_all_goals_from_episode(ep)
                    for g in range(oracle_inds.size):
                        oracle_succ = int(oracle_successes[g])
                        learned_succ = int(learned_successes[g])
                        self.stats_confusion_rew_func[g][oracle_succ].append(learned_succ)

    def infer_from_social_partner_feedbacks(self, feedbacks_str):

        goals_reached_str = feedbacks_str.copy()
        goals_not_reached_str = []
        new_goals_str = []

        if self.params['conditions']['feedback_strategy'] == 'exhaustive':
            # the goals not reached are all the discovered goals minus the set of goals reached

            for ep in range(len(feedbacks_str)):
                # get the list of known goals
                discovered_goals_str = self.goal_sampler.feedback_memory['string'].copy()
                # discovered goals are only the one that are not imagined
                discovered_goals_str = np.array(discovered_goals_str)[np.where(np.array(self.goal_sampler.feedback_memory['imagined']) == 0)].tolist()
                for i in range(len(feedbacks_str[ep])):
                    # if never seen before, add the feedback to the list of known goals
                    if feedbacks_str[ep][i] not in discovered_goals_str:
                        if feedbacks_str[ep][i] not in new_goals_str:
                            new_goals_str.append(feedbacks_str[ep][i])
                    # else remove that feedback from the list, to obtain the complementary set
                    else:
                        index = discovered_goals_str.index(feedbacks_str[ep][i])
                        discovered_goals_str.pop(index)

                goals_not_reached_str.append(discovered_goals_str)

        elif self.params['conditions']['feedback_strategy'] == 'one_pos_one_neg':
            new_goals_per_ep = []
            # the goals not reached are all the discovered goals minus the set of goals reached
            for ep in range(len(feedbacks_str)):
                # get the list of known goals
                discovered_goals_str = self.goal_sampler.feedback_memory['string'].copy()
                # discovered goals are only the one that are not imagined
                discovered_goals_str = np.array(discovered_goals_str)[np.where(np.array(self.goal_sampler.feedback_memory['imagined']) == 0)].tolist()
                new_goals_ep = []
                for i in range(len(feedbacks_str[ep])):
                    # if never seen before, add the feedback to the list of known goals
                    if feedbacks_str[ep][i] not in discovered_goals_str:
                        new_goals_ep.append(feedbacks_str[ep][i])
                        if feedbacks_str[ep][i] not in new_goals_str:
                            new_goals_str.append(feedbacks_str[ep][i])
                    # else remove that feedback from the list, to obtain the complementary set
                    else:
                        index = discovered_goals_str.index(feedbacks_str[ep][i])
                        discovered_goals_str.pop(index)
                new_goals_per_ep.append(new_goals_ep)

                goals_not_reached_str.append(discovered_goals_str)

            # if new goals, add them all to goals reached
            # if no new goals, sample one reached goal
            # sample one not reached goal
            final_goals_reached_str, final_goals_not_reached_str = [], []
            for ep in range(len(feedbacks_str)):
                if len(new_goals_per_ep[ep]) > 0:
                    final_goals_reached_str.append(new_goals_per_ep[ep])
                else:
                    if len(goals_reached_str[ep]) > 0:
                        # sample rare goals in priority (rank-proportional)
                        counts_goal_reached = []
                        for g_str in goals_reached_str[ep]:
                            ind = self.goal_sampler.feedback_memory['string'].index(g_str)
                            counts_goal_reached.append(self.goal_sampler.feedback_memory['reached_counter'][ind])
                        ranks = np.argsort(counts_goal_reached) + 1
                        probas = ranks / ranks.sum()
                        final_goals_reached_str.append([np.random.choice(goals_reached_str[ep], p=probas)])
                    else:
                        final_goals_reached_str.append([])
                if len(goals_not_reached_str[ep]) > 0:
                    final_goals_not_reached_str.append([np.random.choice(goals_not_reached_str[ep])])
                else:
                    final_goals_not_reached_str.append([])
            goals_reached_str = final_goals_reached_str
            goals_not_reached_str = final_goals_not_reached_str
        else:
            raise NotImplementedError

        return goals_reached_str, goals_not_reached_str, new_goals_str

    def clear_eval_history(self):
        for hist in self.evaluation_return_histories:
            hist.clear()

    def convert_feedbacks_to_reward_samples(self,
                                            all_episodes_list,
                                            goals_reached_ids,
                                            goals_not_reached_ids):
        states = []
        goal_id_reached_lists = []
        goal_id_not_reached_lists = []
        for i in range(len(all_episodes_list)):
            states.append(all_episodes_list[i]['obs'][-1])
            goal_id_reached_lists.append(goals_reached_ids[i])
            goal_id_not_reached_lists.append(goals_not_reached_ids[i])

        return states, goal_id_reached_lists, goal_id_not_reached_lists


    def process_evaluation(self, episodes):

        successes = []
        for ep in episodes:
            success = self.oracle_reward_function.eval_goal_from_episode(ep, goal_id=ep['g_id'])  # here g_id is the oracle index directly
            successes.append(success)
            self.evaluation_return_histories[ep['g_id']].append(success)
        return mpi_average(np.mean(successes))
