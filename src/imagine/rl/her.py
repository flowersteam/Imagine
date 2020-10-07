import numpy as np
import time


def make_sample_her_transitions(goal_sampler,
                                goal_invention,
                                p_imagined,
                                rl_positive_ratio,
                                reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        goal_sampler (object): contains the list of discovered goals
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """

    strategy_goal_invention = goal_invention
    p_goal_invention = p_imagined
    ratio_positive = rl_positive_ratio
    n_goals_attempts = 50


    def _sample_her_transitions(episode_batch, goal_ids, batch_size_in_transitions, epoch):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        time_dict = dict()
        goal_ids_len = [len(gid) for gid in goal_ids]
        t_init = time.time()
        T = episode_batch['acts'].shape[1]
        batch_size = batch_size_in_transitions

        # whether to use imagined goals
        goal_invention = strategy_goal_invention
        p_imagined = 0.
        if 'from_epoch' in goal_invention:
            from_epoch = int(goal_invention.split('_')[-1])
            if epoch > from_epoch:
                p_imagined = p_goal_invention

        # find valid buffers (with more than 10 episodes)
        valid_buffers = []
        for i in range(len(goal_ids_len)):
            if goal_ids_len[i] > 0:
                valid_buffers.append(i)

        # sample uniformly in the task buffers, then random episodes from them
        t_sample_ind = time.time()
        if len(valid_buffers) > 0:
            buffer_ids = np.random.choice(valid_buffers, size=batch_size)
            unique, counts = np.unique(buffer_ids, return_counts=True)
            episode_idxs = []
            for i in range(unique.size):
                count = counts[i]
                index_goal = unique[i]
                ids = np.random.randint(goal_ids_len[index_goal], size=count)
                episode_idxs += list(np.array(goal_ids[index_goal])[ids])
        else:
            episode_idxs = np.random.randint(episode_batch['obs'].shape[0], size=batch_size)
        time_dict['time_sample_1'] = time.time() - t_sample_ind
        t_sample_shuffle = time.time()
        np.random.shuffle(episode_idxs)
        time_dict['time_sample_shuffle'] = time.time() - t_sample_shuffle
        t_samples = np.random.randint(T, size=batch_size)
        t_transition_batch = time.time()
        transitions = dict()
        for key in episode_batch.keys():
            if 'g' in key:
                if key != 'g_str':
                    transitions[key] = episode_batch[key][episode_idxs, 0].copy()
                else:
                    transitions[key] = episode_batch[key][episode_idxs].copy()
            else:
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
        time_dict['time_transition_batch'] = time.time() - t_transition_batch
        time_dict['time_sample_ind'] = time.time() - t_sample_ind

        # get list of discovered goals
        discovered_goals_encodings = np.array(goal_sampler.feedback_memory['policy_encoding']).copy()
        discovered_goal_ids = np.array(goal_sampler.feedback_memory['memory_id']).copy()
        all_discovered_goal_ids = np.array(goal_sampler.feedback_memory['memory_id']).copy()
        imagined = np.array(goal_sampler.feedback_memory['imagined']).copy()
        imagined_ind = np.argwhere(imagined==1).flatten()
        not_imagined_ind = np.argwhere(imagined==0).flatten()
        all_perceived_lp = np.array(goal_sampler.perceived_learning_progress).copy()
        perceived_lp = all_perceived_lp.copy()


        nb_discovered_goals = discovered_goal_ids.size

        time_dict['time_pre_replay'] = time.time() - t_init
        time_dict.update(time_reward_func_replay=0,
                         time_argwhere=0,
                         time_random=0)
        t_init = time.time()
        if nb_discovered_goals > 0:
            n_attempts = min(n_goals_attempts, nb_discovered_goals)
            # for all observation, compute the reward with all of the goal.
            # This is done at once to leverage the optimization in numpy
            # sample goal inds to attempt, first from imagined
            n_imagined = min(int(p_imagined * n_attempts), len(discovered_goal_ids[imagined_ind]))
            inds_to_attempt = []
            if n_imagined > 0:
                inds_to_attempt += np.random.choice(discovered_goal_ids[imagined_ind], size=n_imagined, replace=False).tolist()
            n_not_imagined = min(len(not_imagined_ind), n_attempts - n_imagined)
            inds_to_attempt += np.random.choice(discovered_goal_ids[not_imagined_ind], size=n_not_imagined, replace=False).tolist()
            inds_to_attempt = np.array(inds_to_attempt)

            n_attempts = inds_to_attempt.size
            obs = np.repeat(transitions['obs'], n_attempts, axis=0)
            goals = np.tile(inds_to_attempt, batch_size)

            t_ir = time.time()
            rewards = reward_fun(state=obs, goal=goals)[0]
            # print(time.time() - t_ir)
            time_dict['time_reward_func_replay'] += (time.time() - t_ir)

            # figure out where are the positive and negative rewards
            # to balance the ratio of positive vs negative samples
            t_ir = time.time()
            where_neg = (rewards == -1)
            where_pos = (rewards == 0)
            time_dict['time_argwhere'] += (time.time() - t_ir)

            n_positives = int(ratio_positive * batch_size)
            n_negatives = batch_size - n_positives

            t_ir = time.time()
            positives_idx = []
            i = 0
            # try transitions from first to the last find positive rewards.
            # pursue until you found n_positives rewards or you covered the whole batch
            while len(positives_idx) < n_positives and i <= batch_size:
                ind_pos = np.atleast_1d(np.argwhere(where_pos[i * n_attempts: (i + 1) * n_attempts]).squeeze())
                if ind_pos.size > 0:
                    positives_idx.append(i * n_attempts +  np.random.choice(ind_pos))
                i += 1


            # if not enough positives in the whole batch, replace by more negatives
            if len(positives_idx) < n_positives:
                n_negatives = batch_size - len(positives_idx)
                n_positives = len(positives_idx)

            positive_transition_idx = list(np.array(positives_idx) // n_attempts)
            transition_to_search_negatives_in = list(range(batch_size))
            for i in positive_transition_idx:
                transition_to_search_negatives_in.remove(i)
            transition_to_search_negatives_in += positive_transition_idx

            # try transitions from the non positive transitions first,
            # then in the positive transitions as well
            negatives_idx = []
            for i in transition_to_search_negatives_in:
                ind_neg = np.atleast_1d(np.argwhere(where_neg[i * n_attempts: (i + 1) * n_attempts]).squeeze())
                if ind_neg.size > 0:
                    negatives_idx.append(i * n_attempts + np.random.choice(ind_neg))
                if len(negatives_idx) == n_negatives:
                    break

            negatives_idx = np.array(negatives_idx)
            positives_idx = np.array(positives_idx)
            n_replayed = positives_idx.size + negatives_idx.size
            if n_replayed < batch_size:
                ind_transitions_not_replayed = set(range(batch_size)) - set(negatives_idx // n_attempts).union(set(positives_idx // n_attempts))
                ind_transitions_not_replayed = list(ind_transitions_not_replayed)
                if len(ind_transitions_not_replayed) > batch_size - n_replayed:
                    ind_transitions_not_replayed = ind_transitions_not_replayed[:batch_size - n_replayed]
                left = batch_size - len(ind_transitions_not_replayed)  - n_replayed
                if left > 0:
                    ind_transitions_not_replayed += list(np.random.choice(range(batch_size), size=left))
                ind_transitions_not_replayed = np.array(ind_transitions_not_replayed)
            else:
                ind_transitions_not_replayed = np.array([])
                left = 0


            # # # # # # # # # # # # # # # # # # # #
            # Build the batch of transitions
            # # # # # # # # # # # # # # # # # # # #

            # first build an empty dict
            transitions2 = dict()
            for key in transitions.keys():
                shape = list(transitions[key].shape)
                shape[0] = 0
                shape = tuple(shape)
                transitions2[key] = np.array([]).reshape(shape)
            transitions2['r'] = np.array([]).reshape((0,))

            # then add negative samples if there are
            if len(negatives_idx) > 0:
                for key in transitions.keys():
                    if key not in ['g_encoding', 'r', 'g_id']:
                        if 'g' in key:
                            transitions2[key] = np.concatenate([transitions2[key], transitions[key][negatives_idx // n_attempts].copy()], axis=0)
                        else:
                            transitions2[key] = np.concatenate([transitions2[key], transitions[key][negatives_idx // n_attempts, :].copy()], axis=0)
                negative_replay_id = goals[negatives_idx].copy()
                transitions2['g_encoding'] = np.concatenate([transitions2['g_encoding'], discovered_goals_encodings[negative_replay_id]], axis=0)
                transitions2['g_id'] = np.concatenate([transitions2['g_id'], negative_replay_id.copy()], axis=0)
                transitions2['r'] =  np.concatenate([transitions2['r'], - np.ones([len(negatives_idx)])], axis=0)

            if len(positives_idx) > 0:
                for key in transitions.keys():
                    if key not in ['g_encoding', 'r', 'g_id']:
                        if 'g' in key:
                            transitions2[key] = np.concatenate([transitions2[key], transitions[key][positives_idx // n_attempts].copy()], axis=0)
                        else:
                            transitions2[key] = np.concatenate([transitions2[key], transitions[key][positives_idx // n_attempts, :].copy()], axis=0)
                positive_replay_id = goals[positives_idx].copy()
                transitions2['g_encoding'] = np.concatenate([transitions2['g_encoding'], discovered_goals_encodings[positive_replay_id]], axis=0)
                transitions2['g_id'] = np.concatenate([transitions2['g_id'], positive_replay_id.copy()], axis=0)
                transitions2['r'] = np.concatenate([transitions2['r'], np.zeros([len(positives_idx)])], axis=0)

            if len(ind_transitions_not_replayed) > 0:
                for key in transitions.keys():
                    if key not in ['r']:
                        if 'g' in key:
                            transitions2[key] = np.concatenate([transitions2[key], transitions[key][ind_transitions_not_replayed].copy()], axis=0)
                        else:
                            transitions2[key] = np.concatenate([transitions2[key], transitions[key][ind_transitions_not_replayed, :].copy()], axis=0)
                rewards = reward_fun(state=np.atleast_2d(transitions['obs'][ind_transitions_not_replayed]),
                                     goal=np.atleast_1d(transitions['g_id'][ind_transitions_not_replayed]))[0]
                transitions2['r'] = np.concatenate([transitions2['r'], rewards], axis=0)
            # msg = '{} {} {} {} {}'.format(transitions2['obs'].shape[0], len(ind_transitions_not_replayed), len(negatives_idx),
            #                               len(positives_idx), left)
            # logger.info(msg)
            assert transitions2['obs'].shape[0] == batch_size

            ratio_per_goal_in_batch = []
            ind_positive_replay = np.atleast_1d(np.argwhere(transitions2['r'] == 0).squeeze())
            ind_negative_replay = np.atleast_1d(np.argwhere(transitions2['r'] == -1).squeeze())
            for i in range(all_discovered_goal_ids.size):
                g_id = all_discovered_goal_ids[i]
                nb_positive_g_id = np.argwhere(transitions2['g_id'][ind_positive_replay] == g_id).size
                ratio_per_goal_in_batch.append(nb_positive_g_id / batch_size)


            time_dict['time_random'] += (time.time() - t_ir)
            transitions = transitions2

            time_dict['time_replay'] = time.time() - t_init


            t_init2 = time.time()
            # shuffle transitions
            shuffled_inds = np.arange(batch_size)
            np.random.shuffle(shuffled_inds)
            for key in transitions.keys():
                transitions[key] = transitions[key][shuffled_inds].reshape(batch_size, *transitions[key].shape[1:])

        else:
            t_init2 = time.time()
            transitions['r'] = reward_fun(state=transitions['obs'], goal=transitions['g_id'])[0]

        ratio_positive_rewards = (transitions['r']==0).mean()


        time_dict['time_recompute_reward'] = time.time() - t_init2

        assert(transitions['acts'].shape[0] == batch_size_in_transitions)

        lp_scores = all_perceived_lp
        return transitions, ratio_positive_rewards, lp_scores, ratio_per_goal_in_batch, time_dict

    return _sample_her_transitions
