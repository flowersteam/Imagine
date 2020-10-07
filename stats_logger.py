import time
import os
import pickle

import numpy as np
from mpi4py import MPI

from src import logger
from src.utils.util import mpi_average


class StatsLogger:
    def __init__(self, training_worker, evaluation_worker, policy, reward_function, goal_sampler, data_processor, params):
        self.training_worker = training_worker
        self.evaluation_worker = evaluation_worker
        self.policy = policy
        self.reward_function = reward_function
        self.goal_sampler = goal_sampler
        self.data_processor = data_processor
        self.best_success_rate = -1e6
        self.first_time = time.time()
        self.last_time = time.time()
        self.params = params
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.logdir = self.params['experiment_params']['logdir']
        self.nb_goals = len(params['train_descriptions'])
        if self.rank == 0:
            self.latest_policy_path = os.path.join(logger.get_dir(), 'policy_checkpoints/policy_latest.pkl')
            self.best_policy_path = os.path.join(logger.get_dir(), 'policy_checkpoints/policy_best.pkl')
            self.periodic_policy_path = os.path.join(logger.get_dir(), 'policy_checkpoints/policy_{}.pkl')


    def compute_reward_function_metrics(self, epoch, episode_count):
        # Compute and log reward function metrics
        if self.rank == 0:
            if self.params['conditions']['reward_function'] in ['learned_randomforest', 'pretrained_lstm', 'learned_lstm'] :
                save_header = False
                if epoch == 0:
                    save_header = True
                if len(self.reward_function.recent_metrics_record) > 0:
                    with open(os.path.join(self.logdir, 'reward_func_metrics.csv'), 'a') as f:
                        df = self.reward_function.recent_metrics_record[-1].reset_index()
                        df['epoch'] = epoch
                        logger.info(df)
                        df.to_csv(f, header=save_header, index=False)

            # Save stats confusion matrix
            stats_confusion_rew_func = [np.zeros([2, 2]) for _ in range(len(self.params['train_descriptions']))]
            stats_confusion = self.data_processor.stats_confusion_rew_func
            for i in range(len(self.params['train_descriptions'])):
                if len(stats_confusion[i][0]) < 20:
                    stats_confusion_rew_func[i][0, 0] = 0.5
                    stats_confusion_rew_func[i][0, 1] = 0.5
                else:
                    stats_confusion_rew_func[i][0, 0] = 1 - np.mean(stats_confusion[i][0])
                    stats_confusion_rew_func[i][0, 1] = np.mean(stats_confusion[i][0])
                if len(stats_confusion[i][1]) < 20:
                    stats_confusion_rew_func[i][1, 0] = 0.5
                    stats_confusion_rew_func[i][1, 1] = 0.5
                else:
                    stats_confusion_rew_func[i][1, 0] = 1 - np.mean(
                        stats_confusion[i][1])
                    stats_confusion_rew_func[i][1, 1] = np.mean(stats_confusion[i][1])
                for j in range(2):
                    for k in range(2):
                        if np.isnan(stats_confusion_rew_func[i][j, k]):
                            stats_confusion_rew_func[i][j, k] = 0.5
            with open(self.logdir  + 'goal_info/stats_confusion_rew_func_' + str(episode_count) + '.pk', 'wb') as f:
                pickle.dump(stats_confusion_rew_func, f)


    def compute_metrics(self, epoch, episode_count, eval_success_rate, time_logs):

        self.compute_reward_function_metrics(epoch, episode_count)

        # Save observations
        if self.params['experiment_params']['save_obs'] and self.rank == 0:
            with open(self.logdir  + 'obs_' + str(epoch) + '.pk', 'wb') as f:
                pickle.dump(self.data_processor.states_to_save, f)
            self.data_processor.clear_memory_states_to_save()
            with open(self.logdir  + 'goal_discovery_' + str(epoch) + '.pk', 'wb') as f:
                pickle.dump(self.data_processor.goal_sampler.feedback_memory['iter_discovery'], f)

        self.dump_goal_metrics(episode_count)
        logs = self.compute_and_log_interaction_metrics(episode_count, eval_success_rate)

        # record logs
        for key, val in self.policy.logs():
            logger.record_tabular(key, mpi_average(val))

        logger.info(len(logs))
        for key, val in logs:
            logger.record_tabular(key, mpi_average(val))
        logger.record_tabular('pos_rew_ratio', mpi_average(self.policy.get_replay_ratio_positive_reward_stat()))

        if self.rank == 0:
            logger.record_tabular('total_duration (s)', time.time() - self.first_time)
            logger.record_tabular('epoch_duration (s)', time.time() - self.last_time)
            self.last_time = time.time()
            logger.record_tabular('epoch', epoch)
            for key, value in time_logs.time_stats.items():
                logger.record_tabular(key, "{:.3f}".format(value))

            logger.dump_tabular()

        success_rate = mpi_average(eval_success_rate)
        if self.rank == 0:
            # Save the policy if it's better than the previous ones
            self.evaluation_worker.save_policy(self.latest_policy_path)
            if self.params['conditions']['reward_function'] != 'oracle':
                self.reward_function.save_checkpoint(self.params['experiment_params']['logdir'] + 'reward_checkpoints/reward_func_latest_checkpoint')
            if success_rate >= self.best_success_rate:
                self.best_success_rate = success_rate
                logger.info('New best success rate: {}. Saving policy to {} ...'.format(self.best_success_rate, self.best_policy_path))
                self.evaluation_worker.save_policy(self.best_policy_path)
                if self.params['conditions']['reward_function'] != 'oracle':
                    self.reward_function.save_checkpoint(self.params['experiment_params']['logdir'] + 'reward_checkpoints/reward_func_best_checkpoint')
            # Save policy periodically
            if epoch % self.params['experiment_params']['policy_save_interval'] == 0:
                policy_path = self.periodic_policy_path.format(epoch)
                logger.info('Saving periodic policy to {} ...'.format(policy_path))
                self.evaluation_worker.save_policy(policy_path)
                if self.params['conditions']['reward_function'] != 'oracle':
                    self.reward_function.save_checkpoint(self.params['experiment_params']['logdir'] + 'reward_checkpoints/reward_func_checkpoint_{}'.format(str(epoch)))


    def dump_goal_metrics(self, episode_count):
        if self.rank == 0:
            info = dict(discovered_goals=self.goal_sampler.feedback_memory['string'],
                        replay_proba=self.policy.replay_proba,
                        exploration_metrics=self.data_processor.exploration_tracker.metrics.copy()
                        )
            with open(self.logdir+ 'goal_info/info_' + str(episode_count) + '.pk', 'wb') as f:
                pickle.dump(info, f)
            self.data_processor.exploration_tracker.reset_metrics()


    def compute_and_log_interaction_metrics(self, episode_count, eval_success_rate):
        logs = []
        prefix = 'eval'
        logs += [('z_current_eval_success_rate', eval_success_rate)]
        logs += [('episode', episode_count)]
        for i in range(self.nb_goals):
            if len(self.data_processor.evaluation_return_histories[i]) > 0:
                mean = np.mean(self.data_processor.evaluation_return_histories[i])
            else:
                mean = 0
            logs += [(prefix + '/success_goal_' + str(i), mean)]
        logs += [(prefix + '/mean_Q', np.mean(self.evaluation_worker.Q_history))]
        return logs


