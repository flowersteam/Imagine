import argparse
import json
import sys

import numpy as np

sys.path.append('../../../')
from src.utils.util import set_global_seeds
import src.imagine.experiment.config as config
from src.imagine.interaction import RolloutWorker
from src.imagine.goal_sampler import GoalSampler
from src.playground_env.reward_function import get_reward_from_state
from src.playground_env.descriptions import generate_all_descriptions


def get_params_for_notebook(path):
    PARAMS_FILE = path + 'params.json'
    with open(PARAMS_FILE) as json_file:
        params = json.load(json_file)

    env = 'PlaygroundNavigationRender-v1'
    seed = np.random.randint(1e6)

    params, rank_seed = config.configure_everything(rank=0,
                                                    seed=seed,
                                                    num_cpu=params['experiment_params']['n_cpus'],
                                                    env=env,
                                                    trial_id=0,
                                                    n_epochs=10,
                                                    reward_function=params['conditions']['reward_function'],
                                                    policy_encoding=params['conditions']['policy_encoding'],
                                                    feedback_strategy=params['conditions']['feedback_strategy'],
                                                    policy_architecture=params['conditions']['policy_architecture'],
                                                    goal_invention=params['conditions']['goal_invention'],
                                                    reward_checkpoint=params['conditions']['reward_checkpoint'],
                                                    rl_positive_ratio=params['conditions']['rl_positive_ratio'],
                                                    p_partner_availability=params['conditions'][
                                                        'p_social_partner_availability'],
                                                    imagination_method=params['conditions']['imagination_method'],
                                                    git_commit='',
                                                    display=False)
    return params


def get_modules_for_notebook(path, params):
    EPOCH = '160'
    POLICY_FILE = path + 'policy_checkpoints/policy_{}.pkl'.format(EPOCH)
    policy_language_model, reward_language_model = config.get_language_models(params)

    onehot_encoder = config.get_one_hot_encoder(params['all_descriptions'])
    # Define the goal sampler for training
    goal_sampler = GoalSampler(policy_language_model=policy_language_model,
                               reward_language_model=reward_language_model,
                               goal_dim=policy_language_model.goal_dim,
                               one_hot_encoder=onehot_encoder,
                               params=params)

    reward_function = config.get_reward_function(goal_sampler, params)
    if params['conditions']['reward_function'] == 'learned_lstm':
        reward_function.restore_from_checkpoint(path + 'reward_checkpoints/reward_func_checkpoint_{}'.format(EPOCH))
    policy_language_model.set_reward_function(reward_function)
    if reward_language_model is not None:
        reward_language_model.set_reward_function(reward_function)
    goal_sampler.update_discovered_goals(params['all_descriptions'], episode_count=0, epoch=0)

    # Define learning algorithm
    policy = config.configure_learning_algo(reward_function=reward_function,
                                            goal_sampler=goal_sampler,
                                            params=params)

    policy.load_params(POLICY_FILE)
    return policy_language_model, reward_language_model, policy, reward_function, goal_sampler