import os
import sys
import time
import argparse

os.environ["MKL_NUM_THREADS"] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from mpi4py import MPI

sys.path.append('../../../')
sys.path.append('../../../../')
from src.imagine.social_partner import SocialPartner
from src import logger
from src.imagine.experiment import config
from src.imagine.interaction import RolloutWorker
from src.utils.util import fork, get_time_tracker
from src.imagine.goal_sampler import GoalSampler, EvalGoalSampler
from src.imagine.data_processor import DataProcessor
from src.stats_logger import StatsLogger

NUM_CPU = 1
NB_EPOCHS = 500
ENV = 'PlaygroundNavigation-v1'
IMAG_METHOD = 'CGH'
REW_FUNC = 'learned_lstm'
POLICY_ARCHITECTURE = 'modular_attention'
POLICY_ENCODING = 'lstm'
GOAL_INVENTION = 'from_epoch_10'
SOCIAL_STRATEGY = 'exhaustive'
P_PARTNER_AVAIL = 1
RL_RATIO_POSITIVE = 0.5
REWARD_CHECKPOINT = 'model_0.pk' # 'pre_trained/reward_func_checkpoint_270'


def train(policy, training_worker, evaluation_worker, data_processor, goal_sampler, eval_goal_sampler, reward_function,
          n_epochs, n_test_rollouts, n_cycles, n_batches, social_partner,
          stats_logger, params, **kwargs):
    print('\n\n')
    print(params['conditions'])
    print('\n\n')

    logger.info("Training...")
    rank = MPI.COMM_WORLD.Get_rank()
    episode_count = 0

    for epoch in range(n_epochs):

        # # # # # # # # # # # # # # # # #
        # TRAIN
        # # # # # # # # # # # # # # # # # #

        # Prepare epoch
        time_tracker = get_time_tracker()  # dict containing time statistics
        init_time = time.time()
        training_worker.clear_history()
        if rank == 0:
            logger.info('\n \t \t Epoch {}, {} goals discovered'.format(epoch, goal_sampler.nb_discovered_goals))

        # Loop over cycles
        for i_c in range(n_cycles):
            # # # # # #
            # Sample goals
            # # # # # #
            timee = time.time()
            exploit, goals_str, goals_encodings, goals_ids, imagined = goal_sampler.sample_targets(epoch)
            time_tracker.add(time_sample_goal=time.time() - timee)

            # # # # # #
            # Interact with the environment
            # # # # # #
            timee = time.time()
            episodes = training_worker.generate_rollouts(exploit=exploit,
                                                         imagined=imagined,
                                                         goals_str=goals_str,
                                                         goals_encodings=goals_encodings,
                                                         goals_ids=goals_ids)
            episode_count += params['experiment_params']['rollout_batch_size'] * params['experiment_params']['n_cpus']
            time_tracker.add(time_rollout=time.time() - timee)

            # # # # # #
            # Interact with the social partner
            # # # # # #

            # See whether social partner is available
            partner_available = social_partner.is_available()
            # Ask for social partner's feedbacks
            # if not available, do not provide feedback_str
            # but keep records of train, test, extra descr for statistics
            timee = time.time()
            feedbacks_str, train_descr, test_descr, extra_descr = social_partner.get_feedback(episodes)
            if not partner_available:
                feedbacks_str = None
            time_tracker.add(time_social_partner=time.time() - timee)


            # # # # # #
            # Process data
            # # # # # #
            timee = time.time()
            reward_data, goals_reached_ids, time_dict = data_processor.process(episode_count,
                                                                               epoch,
                                                                               episodes,
                                                                               partner_available,
                                                                               feedbacks_str,
                                                                               train_descr,
                                                                               test_descr,
                                                                               extra_descr
                                                                               )
            time_tracker.add(time_data_process=time.time() - timee)
            time_tracker.add(**time_dict)

            # # # # # #
            # Dispatch data to the learning algorithms: policy and reward function
            # # # # # #
            timee = time.time()
            policy.store_episode(episodes, goals_reached_ids)
            time_tracker.add(time_store_policy=time.time() - timee)
            timee = time.time()
            reward_function.store(reward_data)
            time_tracker.add(time_store=time.time() - timee)

            # # # # # #
            # Update the policy and critic
            # # # # # #
            timee = time.time()
            if epoch > 0 or params['conditions']['reward_function'] == 'oracle':
                for _ in range(n_batches):
                    _, _, times_training = policy.train(epoch)
                    time_tracker.add(**times_training)
                policy.update_target_net()
            time_tracker.add(time_train=time.time() - timee)

        # # # # # #
        # Update the reward function (and language model)
        # # # # # #
        timee = time.time()
        reward_function.update(epoch)
        reward_function.share_reward_function_to_all_cpus()
        time_tracker.add(time_reward_func_update=time.time() - timee)

        # # # # # # # # # # # # # # # # # #
        # EVALUATE
        # # # # # # # # # # # # # # # # # #
        evaluation_worker.clear_history()
        eval_goal_sampler.reset()
        data_processor.clear_eval_history()
        timee = time.time()
        eval_episodes = []

        for _ in range(n_test_rollouts):
            # Sample goal for evaluation
            exploit, goals_str, goals_encodings, goals_ids = eval_goal_sampler.sample(method=params['experiment_params']['method_test'])
            # Run evaluation rollouts
            episodes = evaluation_worker.generate_rollouts(exploit=exploit,
                                                           imagined=False,
                                                           goals_str=goals_str,
                                                           goals_encodings=goals_encodings,
                                                           goals_ids=goals_ids)
            eval_episodes += episodes
        eval_success_rate = data_processor.process_evaluation(eval_episodes)
        time_tracker.add(time_eval=time.time() - timee,
                         time_epoch=time.time() - init_time)

        # # # # # # # # # # # # # # # # # #
        # LOGS
        # # # # # # # # # # # # # # # # # #
        stats_logger.compute_metrics(epoch, episode_count, eval_success_rate, time_tracker)


def launch(**kwargs):
    # Fork for multi-CPU MPI implementation.
    rank = fork(kwargs['num_cpu'])

    # Configure everything and log parameters
    params, rank_seed = config.configure_everything(rank, **kwargs)

    # Define language model
    policy_language_model, reward_language_model = config.get_language_models(params)

    # Define the one-hot_encoder
    onehot_encoder = config.get_one_hot_encoder(params['train_descriptions'] + params['test_descriptions'])

    # Define the goal sampler for training
    goal_sampler = GoalSampler(policy_language_model=policy_language_model,
                               reward_language_model=reward_language_model,
                               goal_dim=policy_language_model.goal_dim,
                               one_hot_encoder=onehot_encoder,
                               params=params)

    # Define reward function
    reward_function = config.get_reward_function(goal_sampler=goal_sampler,
                                                 params=params)
    oracle_reward_function = config.get_oracle_reward_function(goal_sampler, params)

    policy_language_model.set_reward_function(reward_function)
    if reward_language_model is not None:
        reward_language_model.set_reward_function(reward_function)

    # Define the goal sampler for evaluation
    eval_goal_sampler = EvalGoalSampler(policy_language_model=policy_language_model,
                                        one_hot_encoder=onehot_encoder,
                                        params=params)
    # Give reward function to goal sampler to track metrics
    goal_sampler.store_reward_function(reward_function)

    # Define learning algorithm
    policy = config.configure_learning_algo(reward_function=reward_function,
                                            goal_sampler=goal_sampler,
                                            params=params)

    # Define the social partner
    social_partner = SocialPartner(oracle_reward_function=oracle_reward_function,
                                   **params['social_partner_params'],
                                   params=params)

    # Define the data processor
    data_processor = DataProcessor(reward_function=reward_function,
                                   oracle_reward_function=oracle_reward_function,
                                   goal_sampler=goal_sampler,
                                   params=params)

    # Define the worker to interact with the environment (training and evaluation)
    training_worker = RolloutWorker(make_env=params['make_env'],
                                    policy=policy,
                                    reward_function=reward_function,
                                    params=params,
                                    **params['training_rollout_params'])
    training_worker.seed(rank_seed)

    evaluation_worker = RolloutWorker(make_env=params['make_env'],
                                      policy=policy,
                                      reward_function=reward_function,
                                      params=params,
                                      **params['evaluation_rollout_params'],
                                      render=False)
    evaluation_worker.seed(rank_seed * 10)

    stats_logger = StatsLogger(goal_sampler=goal_sampler,
                               data_processor=data_processor,
                               training_worker=training_worker,
                               evaluation_worker=evaluation_worker,
                               reward_function=reward_function,
                               policy=policy,
                               params=params)

    train(logdir=params['experiment_params']['logdir'],
          policy=policy,
          training_worker=training_worker,
          goal_sampler=goal_sampler,
          eval_goal_sampler=eval_goal_sampler,
          evaluation_worker=evaluation_worker,
          social_partner=social_partner,
          n_epochs=params['experiment_params']['n_epochs'],
          n_test_rollouts=params['experiment_params']['n_test_rollouts'],
          n_cycles=params['experiment_params']['n_cycles'],
          n_batches=params['experiment_params']['n_batches'],
          reward_function=reward_function,
          stats_logger=stats_logger,
          data_processor=data_processor,
          params=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--env', type=str, default=ENV, help='the name of the OpenAI Gym environment that you want to train on')
    add('--trial_id', type=int, default='333', help='trial identifier, name of the saving folder')
    add('--num_cpu', type=int, default=NUM_CPU, help='the number of CPU cores to use (using MPI)')
    add('--n_epochs', type=int, default=NB_EPOCHS, help='the number of training epochs to run')
    add('--reward_function', type=str, default=REW_FUNC, help="reward function to use, 'learned', 'oracle' or 'pretrained'")
    add('--seed', type=int, default=np.random.randint(int(1e6)), help='the random seed used to seed both the environment and the training code')
    add('--feedback_strategy', type=str, default=SOCIAL_STRATEGY, help="Strategy to use to provide positive feedback to the classifier 'exhaustive', 'one_pos_one_neg'")
    add('--git_commit', default='', type=str, help="git commit")
    add('--rl_positive_ratio', default=RL_RATIO_POSITIVE, type=str, help="ratio of positive samples per instruction")
    add('--policy_architecture', default=POLICY_ARCHITECTURE,  type=str,  help="'modular_attention', 'flat_concat', 'flat_attention'")
    add('--policy_encoding', default=POLICY_ENCODING,  type=str,  help="'glove', 'one_hot', 'lstm'")
    add('--goal_invention', default=GOAL_INVENTION,  type=str,  help="'from_epoch_x")
    add('--reward_checkpoint', default=REWARD_CHECKPOINT,  type=str,  help="reward checkpoint file for pretrained")
    add('--p_partner_availability', default=P_PARTNER_AVAIL,  type=float,  help="probability availability partner")
    add('--imagination_method', default=IMAG_METHOD,  type=str,  help="CGH, low_precision, low_coverage, oracle, random")
    kwargs = vars(parser.parse_args())
    launch(**kwargs)
