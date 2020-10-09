from collections import deque
import numpy as np
import pandas as pd
from mpi4py import MPI

import torch
from src import logger
from src.utils.reward_func_util import StateIdBuffer, Batch
from src.utils.reward_func_util import get_metrics_by_instructions_lstm
from src.imagine.reward_function.model import RewardFunctionModel


class RewardFunctionLSTM:
    def __init__(self, goal_sampler, params):
        '''
        The reward function object used to train the reward function model alongside the policy.
        The reward function is trained from goals sampled by the goal sampler and data collected by the policy.

        Parameters
        ----------
        goal_sampler : A goal smapler object (use to access language information about goal descriptions)
        params : contains a variety of params:
            batch_size: Size of batch use to compute gradient before backpropagating.
            positive_ratio: Ration between positive and negative reward labels used to construct batches during training.
            n_epoch: maximum number of epochs done at each reward function update.
            n_batch: number of batches per epoch.
            freq_update: frequeny of update of the reward function (computed with respect to policy updates).
            proportion_split_test: Number of samples saved to construct test set to self evaluate the reward function's
            generalization capabilities.

        '''
        # Learning Meta Parameters
        self.batch_size = params['reward_function']['batch_size']
        self.positive_ratio = params['reward_function']['reward_positive_ratio']
        self.n_epoch = params['reward_function']['max_n_epoch']
        self.n_batch = params['reward_function']['n_batch']
        self.params = params
        self.freq_update = params['reward_function']['freq_update']
        self.early_stopping = params['reward_function']['early_stopping']
        self.proportion_split_test = 0.2
        self.n_added_states = 0
        # Environment params read from params
        inds_obj = params['dims']['inds_objs']
        obj_size = len(inds_obj[0])
        body_size = int(inds_obj[0][0])
        n_obj = len(inds_obj)
        state_size = params['dims']['obs']
        self.goal_sampler = goal_sampler
        self.reward_function = RewardFunctionModel(or_params_path=params['or_params_path'],
                                                   body_size=body_size, obj_size=obj_size, n_obj=n_obj,
                                                   state_size=state_size,
                                                   voc_size=self.goal_sampler.one_hot_encoder.vocab.size,
                                                   sequence_length=goal_sampler.one_hot_encoder.max_seq_length,
                                                   batch_size=self.batch_size,
                                                   num_hidden_lstm=params['reward_function']['num_hidden_lstm'])

        self.optimizer = torch.optim.Adam(params=self.reward_function.get_params(),
                                          lr=params['reward_function']['learning_rate'])

        self.size_encoding = self.reward_function.num_hidden_lstm
        self.nb_max_added_states = self.params['experiment_params']['n_cpus'] * \
                                   self.params['experiment_params']['n_cycles'] * \
                                   self.params['experiment_params']['rollout_batch_size']

        self.rank = MPI.COMM_WORLD.Get_rank()
        self.state_id_buffer = StateIdBuffer(max_len=50000)

        self.recent_metrics_record = deque(maxlen=20)
        self.metrics = pd.DataFrame()

    def get_or_output(self, input):
        return self.reward_function.or_model(input)

    def store(self, data):
        if self.rank == 0 and data is not None:
            states, goal_reached_ids_list, goal_not_reached_ids_list = data

            self.n_added_states += len(states)
            for state, goal_reach_ids, goal_not_reach_ids in zip(states, goal_reached_ids_list,
                                                                 goal_not_reached_ids_list):
                self.state_id_buffer.update(state, goal_reach_ids, goal_not_reach_ids)

    def get_count_per_goal_id(self):
        counts = []
        ids = []
        for k, buffs in self.state_id_buffer.state_id_buffer.items():
            ids.append(k)
            counts.append(len(buffs['pos_reward']))
        return np.array(counts), np.array(ids)

    def get_goal_embedding(self, goal_ids):
        return self.goal_sampler.feedback_encodings_memory[goal_ids]

    def save_params(self, path):
        with open(path, 'wb') as f:
            torch.save(self.reward_function.state_dict(), f)

    def load_params(self, path):
        with open(path, 'rb') as f:
            self.reward_function.load_state_dict(torch.load(f))

    def update(self, epoch):
        if self.rank == 0:
            # Extract the 10 percent of the freshly added states in order to perform testing
            states_train, state_id_buffer_train, states_test, state_id_buffer_test = self.state_id_buffer.split_buffer(
                int(min(self.n_added_states, self.proportion_split_test * self.nb_max_added_states)))

            if epoch % self.freq_update == 0:

                batch = Batch(states_train, state_id_buffer_train, self.reward_function.batch_size,
                              self.goal_sampler.id2one_hot, self.positive_ratio)

                count_since_cost_improvement = 0

                best_cost = 1e6
                best_f1 = -1
                for e in range(self.n_epoch):

                    losses_over_epoch = []
                    for b in range(self.n_batch):
                        batch_s, batch_descr, batch_r = batch.next_batch()
                        batch_s = torch.tensor(batch_s, dtype=torch.float32)
                        batch_descr = torch.tensor(batch_descr, dtype=torch.float32)
                        batch_r = torch.tensor(batch_r, dtype=torch.float32)
                        batch_pred = self.reward_function(batch_s, batch_descr).squeeze()
                        # todo batch are empty when there is no negatives (biased policy at first)

                        if batch_s.size != 0:
                            loss = torch.mean(
                                -torch.log(batch_pred) * batch_r + (1 - batch_r) * (-torch.log(1 - batch_pred)))
                            losses_over_epoch.append(float(loss))
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                    mean_loss = np.mean(losses_over_epoch)

                    if batch_s.size != 0:
                        f1 = 0
                        if self.early_stopping == 'f1':
                            new_metrics = self.evaluate(states_test, state_id_buffer_test)
                            f1 = new_metrics['f1_score']['model']
                            if f1 > best_f1:
                                count_since_cost_improvement = 0
                                self.save_checkpoint(self.params['experiment_params']['logdir'] + 'tmp/best_checkpoint')
                                best_f1 = f1
                            else:
                                count_since_cost_improvement += 1
                                if count_since_cost_improvement == 15:
                                    break
                        elif self.early_stopping == 'cost':
                            if mean_loss < best_cost - (best_cost * 0.1):
                                best_cost = mean_loss
                                count_since_cost_improvement = 0
                            else:
                                count_since_cost_improvement += 1
                                if count_since_cost_improvement == 15:
                                    break
                        logger.info("Epoch: ", e + 1, '/', self.n_epoch, " Cost: ", mean_loss)

                if self.early_stopping == 'f1':
                    self.restore_from_checkpoint(self.params['experiment_params']['logdir'] + 'tmp/best_checkpoint')
                # save current checkpoint so that it can be loaded by or_module cpus
                self.save_checkpoint(self.params['experiment_params']['logdir'] + 'tmp/current_checkpoint')

            new_metrics = self.evaluate(states_test, state_id_buffer_test)
            self.recent_metrics_record.append(new_metrics)
            self.metrics = pd.concat(self.recent_metrics_record).groupby('instruction').mean().round(2)

        self.metrics = MPI.COMM_WORLD.bcast(self.metrics, root=0)
        self.n_added_states = 0

    def save_checkpoint(self, path):
        with open(path, 'wb') as f:
            torch.save(self.reward_function.state_dict(), f)

    def restore_from_checkpoint(self, path):
        with open(path, 'rb') as f:
            self.reward_function.load_state_dict(torch.load(f))
        self.goal_sampler.update_embeddings()

    def share_reward_function_to_all_cpus(self):
        # wait for all cpus
        MPI.COMM_WORLD.barrier()
        self.restore_from_checkpoint(self.params['experiment_params']['logdir'] + 'tmp/current_checkpoint')

    def _generate_test_set_for_instruction(self, state_id_buffer, states_test, instruction_id):
        states_test_out = []
        goal_ids_test_out = []
        rewards_test = []
        for state_id, reward in state_id_buffer[instruction_id]:
            states_test_out.append(states_test[state_id])
            goal_ids_test_out.append(instruction_id)
            rewards_test.append(reward)
        return states_test_out, goal_ids_test_out, rewards_test

    def evaluate(self, states_test, state_id_buffer_test):

        if len(states_test) > 0:
            s_test = []
            goal_ids_test = []
            r_test = []
            for id in state_id_buffer_test.keys():
                s_test_id, goal_ids_test_id, r_test_id = self._generate_test_set_for_instruction(state_id_buffer_test,
                                                                                                 states_test, id)
                s_test.extend(s_test_id)
                goal_ids_test.extend(goal_ids_test_id)
                r_test.extend(r_test_id)

            goal_ids_test = np.array(goal_ids_test)
            new_metrics = get_metrics_by_instructions_lstm(r_test,
                                                           goal_ids_test,
                                                           predict_func=self.predict,
                                                           data=s_test)
        else:
            new_metrics = self.recent_metrics_record[-1]

        return new_metrics

    def predict(self, state, goal_ids):

        goal_ids = goal_ids.flatten().astype(np.int)
        descriptions_embedding = np.atleast_2d(np.array(self.goal_sampler.feedback_memory['reward_encoding'])[goal_ids])
        self.reward_function.eval()
        with torch.no_grad():
            predictions = torch.round(self.reward_function.pred_from_h_lstm(torch.tensor(state, dtype=torch.float32),
                                                                torch.tensor(descriptions_embedding,
                                                                             dtype=torch.float32)))
        return np.atleast_1d(predictions.squeeze()).astype(np.int) - 1, None

    def eval_all_goals_from_episode(self, episode):
        discovered_ids = np.array(self.goal_sampler.feedback_memory['memory_id'])
        obs = np.repeat(episode['obs'][-1].reshape(1, -1), discovered_ids.size, axis=0)
        rewards = self.predict(state=obs, goal_ids=discovered_ids)[0]
        successes = rewards + 1
        return successes

    def get_metrics_record(self):
        return self.recent_metrics_record