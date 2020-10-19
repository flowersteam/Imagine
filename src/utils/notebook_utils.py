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
    EPOCH = '160'
    POLICY_FILE = path + 'policy_checkpoints/policy_{}.pkl'.format(EPOCH)
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
                                                    git_commit='')
    return params