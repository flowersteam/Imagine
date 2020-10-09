import numpy as np
from mpi4py import MPI
from src.imagine.goal_generator.simple_sentence_generator import SentenceGeneratorHeuristic
from src import logger

class GoalSampler:
    def __init__(self,
                 policy_language_model,
                 reward_language_model,
                 goal_dim,
                 one_hot_encoder,
                 params):

        self.policy_language_model = policy_language_model
        self.reward_language_model = reward_language_model
        self.goal_dim = goal_dim
        self.params = params

        self.nb_feedbacks = 0
        self.nb_positive_feedbacks = 0
        self.nb_negative_feedbacks = 0

        self.feedback2id = dict()
        self.id2feedback = dict()
        self.id2oracleid = dict()
        self.feedback2one_hot = dict()
        self.id2one_hot = dict()
        self.feedback_memory = dict(memory_id=[],
                                    string=[],
                                    iter_discovery=[],
                                    target_counter=[],
                                    reached_counter=[],
                                    oracle_id=[],
                                    f1_score=[],
                                    policy_encoding=[],
                                    reward_encoding=[],
                                    imagined=[],
                                    )
        self.imagined_goals = dict(string=[],
                                   competence=[],
                                   lp=[])
        self.one_hot_encoder = one_hot_encoder
        self.goal_generator = SentenceGeneratorHeuristic(params['train_descriptions'],
                                                         params['test_descriptions'],
                                                         sentences=None,
                                                         method=params['conditions']['imagination_method'])
        self.nb_discovered_goals = 0
        self.score_target_goals = None
        self.perceived_learning_progress = None
        self.perceived_competence = None
        self.feedback_stats = None
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.num_cpus = params['experiment_params']['n_cpus']
        self.rollout_batch_size = params['experiment_params']['rollout_batch_size']

        self.not_imagined_goal_ids = np.array([])
        self.imagined_goal_ids = np.array([])

    def store_reward_function(self, reward_function):
        self.reward_function = reward_function

    def update_embeddings(self):
        # embeddings must be updated when the language model is udpated
        for i, goal_str in enumerate(self.feedback_memory['string']):
            if self.reward_language_model is not None:
                reward_encoding = self.reward_language_model.encode(goal_str)
                self.feedback_memory['reward_encoding'][i] = reward_encoding.copy()
            policy_encoding = self.policy_language_model.encode(goal_str)
            self.feedback_memory['policy_encoding'][i] = policy_encoding.copy()


    def add_entries_to_feedback_memory(self, str_list, episode_count, imagined):
        for goal_str in str_list:
            if goal_str not in self.feedback2id.keys():
                memory_id = self.nb_discovered_goals
                if goal_str in self.params['train_descriptions']:
                    oracle_id = self.params['train_descriptions'].index(goal_str)
                else:
                    oracle_id = None
                one_hot = self.one_hot_encoder.encode(goal_str.lower().split(" "))
                self.feedback2one_hot[goal_str] = one_hot
                self.id2one_hot[memory_id] = one_hot
                if self.reward_language_model is not None:
                    reward_encoding = self.reward_language_model.encode(goal_str)
                    self.feedback_memory['reward_encoding'].append(reward_encoding.copy())
                policy_encoding = self.policy_language_model.encode(goal_str)
                self.feedback2id[goal_str] = memory_id
                self.id2oracleid[memory_id] = oracle_id
                self.id2feedback[memory_id] = goal_str
                self.feedback_memory['memory_id'].append(memory_id)
                self.feedback_memory['oracle_id'].append(oracle_id)
                self.feedback_memory['string'].append(goal_str)
                self.feedback_memory['target_counter'].append(0)
                self.feedback_memory['reached_counter'].append(0)
                self.feedback_memory['iter_discovery'].append(episode_count)
                self.feedback_memory['f1_score'].append(0)
                self.feedback_memory['policy_encoding'].append(policy_encoding.copy())
                self.feedback_memory['imagined'].append(imagined)
                self.nb_discovered_goals += 1
            elif goal_str in self.feedback2id.keys() and not imagined:  # if goal previously imagined is discovered later, change its status
                ind = self.feedback_memory['string'].index(goal_str)
                if self.feedback_memory['imagined'][ind] == 1:
                    self.feedback_memory['imagined'][ind] = 0
                    logger.info('Goal already imagined:', goal_str)


    def update_discovered_goals(self,
                                new_goals_str,
                                episode_count,
                                epoch):
        # only done in cpu 0
        self.add_entries_to_feedback_memory(str_list=new_goals_str,
                                            episode_count=episode_count,
                                            imagined=0)

        # Decide whether to generate new goals
        goal_invention = self.params['conditions']['goal_invention']
        imagined = False
        if 'from_epoch' in goal_invention:
            from_epoch = int(goal_invention.split('_')[-1])
            if epoch > from_epoch:
                imagined = True
        if len(new_goals_str) > 0 and imagined:
            new_imagined_goals = []
            inds_not_imagined = np.argwhere(np.array(self.feedback_memory['imagined']) == 0).flatten()
            self.goal_generator.update_model(np.array(self.feedback_memory['string'])[inds_not_imagined])
            generated_goals = self.goal_generator.generate_sentences(n='all')
            for gen_g in generated_goals:
                if gen_g not in self.imagined_goals['string']:
                    self.imagined_goals['string'].append(gen_g)
                    self.imagined_goals['competence'].append(0)
                    self.imagined_goals['lp'].append(0)
                    new_imagined_goals.append(gen_g)
            self.add_entries_to_feedback_memory(str_list=new_imagined_goals,
                                                episode_count=episode_count,
                                                imagined=1)


    def update(self,
               current_episode,
               all_episodes,
               partner_available,
               goals_reached_str,
               goals_not_reached_str):

        imagined_inds = np.argwhere(np.array(self.feedback_memory['imagined']) == 1).flatten()
        not_imagined_inds = np.argwhere(np.array(self.feedback_memory['imagined']) == 0).flatten()
        self.not_imagined_goal_ids = np.array(self.feedback_memory['memory_id'])[not_imagined_inds]
        self.imagined_goal_ids = np.array(self.feedback_memory['memory_id'])[imagined_inds]

        # only done in cpu 0
        n_episodes = len(all_episodes)
        attempted_goals_ids = []
        exploit = []
        for ep in all_episodes:
            exploit.append(ep['exploit'])
            attempted_goals_ids.append(ep['g_id'])

        if partner_available:
            # if partner is available, simply encodes what it said
            assert n_episodes == len(goals_reached_str) == len(goals_not_reached_str) == len(exploit) == len(attempted_goals_ids)
            # Get indexes in the order of discovery of the attempted goals, reached_goals, not reached_goals
            goals_reached_ids = []
            goals_not_reached_ids = []
            for i in range(n_episodes):
                goals_reached_ids.append([])
                goals_not_reached_ids.append([])
                for goal_str in goals_reached_str[i]:
                    goals_reached_ids[-1].append(self.feedback2id[goal_str])
                for goal_str in goals_not_reached_str[i]:
                    goals_not_reached_ids[-1].append(self.feedback2id[goal_str])
        else:
            goals_reached_ids = []
            goals_not_reached_ids = []
            final_obs = np.array([ep['obs'][-1] for ep in all_episodes])
            # test 50 goals for each episode
            discovered_goal_ids = np.array(self.feedback_memory['memory_id'])
            not_imagined_ind = np.argwhere(np.array(self.feedback_memory['imagined']) == 0).flatten()
            discovered_goal_ids = discovered_goal_ids[not_imagined_ind]
            n_attempts = min(50, len(discovered_goal_ids))
            goals_to_try = np.random.choice(discovered_goal_ids, size=n_attempts, replace=False)
            obs = np.repeat(final_obs, n_attempts, axis=0)
            goals = np.tile(goals_to_try, final_obs.shape[0])
            rewards = self.reward_function.predict(state=obs, goal_ids=goals)[0]

            for i in range(len(all_episodes)):
                pos_goals = goals_to_try[np.where(rewards[i * n_attempts: (i + 1) * n_attempts] == 0)].tolist()
                goals_reached_ids.append(pos_goals)
                neg_goals = goals_to_try[np.where(rewards[i * n_attempts: (i + 1) * n_attempts] == -1)].tolist()
                goals_not_reached_ids.append(neg_goals)

        return goals_reached_ids, goals_not_reached_ids


    def share_info_to_all_cpus(self):

        # share data across cpus
        self.feedback_memory = MPI.COMM_WORLD.bcast(self.feedback_memory, root=0)
        self.feedback2id = MPI.COMM_WORLD.bcast(self.feedback2id, root=0)
        self.id2oracleid = MPI.COMM_WORLD.bcast(self.id2oracleid, root=0)
        self.id2feedback = MPI.COMM_WORLD.bcast(self.id2feedback, root=0)
        self.feedback2one_hot = MPI.COMM_WORLD.bcast(self.feedback2one_hot, root=0)
        self.nb_discovered_goals = MPI.COMM_WORLD.bcast(self.nb_discovered_goals, root=0)
        self.imagined_goals = MPI.COMM_WORLD.bcast(self.imagined_goals, root=0)
        self.one_hot_encoder = MPI.COMM_WORLD.bcast(self.one_hot_encoder, root=0)



    def sample_targets(self, epoch):
        """
        Sample targets for all cpus and all batch, then scatter to the different cpus
        """
        # Decide whether to exploit or not
        exploit = True if np.random.random() < 0.1 else False
        strategy = 'random'

        goal_invention = self.params['conditions']['goal_invention']
        imagined = False
        if 'from_epoch' in goal_invention:
            from_epoch = int(goal_invention.split('_')[-1])
            if epoch > from_epoch:
                imagined = np.random.random() < self.params['conditions']['p_imagined']

        if self.rank == 0:
            all_goals_str = []
            all_goals_encodings = []
            all_goals_ids = []

            for i in range(self.num_cpus):
                goals_str = []
                goals_encodings = []
                goals_ids = []
                for j in range(self.rollout_batch_size):
                    # when there is no goal in memory, sample random goal from standard normal distribution
                    if len(self.feedback_memory['memory_id']) == 0:
                            goals_encodings.append(np.random.normal(size=self.goal_dim))
                            goals_str.append('Random Goal')
                            goals_ids.append(-1)
                    else:
                        if strategy == 'random':
                            if imagined and self.imagined_goal_ids.size > 0:
                                ind = np.random.choice(self.imagined_goal_ids)
                            else:
                                ind = np.random.choice(self.not_imagined_goal_ids)
                        else:
                            raise NotImplementedError
                        goals_encodings.append(self.feedback_memory['policy_encoding'][ind])
                        goals_str.append(self.id2feedback[ind])
                        goals_ids.append(ind)
                all_goals_str.append(goals_str)
                all_goals_encodings.append(goals_encodings)
                all_goals_ids.append(goals_ids)
        else:
            all_goals_str = []
            all_goals_encodings = []
            all_goals_ids = []

        goals_str = MPI.COMM_WORLD.scatter(all_goals_str, root=0)
        goals_encodings = MPI.COMM_WORLD.scatter(all_goals_encodings, root=0)
        goals_ids = MPI.COMM_WORLD.scatter(all_goals_ids, root=0)

        return exploit, goals_str, goals_encodings, goals_ids, imagined


class EvalGoalSampler:

    def __init__(self, policy_language_model, one_hot_encoder, params):
        self.descriptions = params['train_descriptions']
        self.nb_descriptions = len(self.descriptions)
        self.count = 0
        self.policy_language_model = policy_language_model
        self.rollout_batch_size = params['evaluation_rollout_params']['rollout_batch_size']
        self.params = params

    def reset(self):
        self.count = 0

    def sample(self, method='robin'):
        # print(self.descriptions[self.count])
        goals_str = []
        goals_encodings = []
        goals_ids = []

        if method == 'robin':
            ind = self.count
        elif method == 'random':
            ind = np.random.randint(self.nb_descriptions)
        else:
            raise NotImplementedError

        for _ in range(self.rollout_batch_size):
            g_str = self.descriptions[ind]
            goals_str.append(g_str)
            policy_encoding = self.policy_language_model.encode(g_str).flatten()
            goals_encodings.append(policy_encoding)
            goals_ids.append(ind)
        self.count += 1
        return True, goals_str, goals_encodings, goals_ids
