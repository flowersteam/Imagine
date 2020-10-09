import os
import datetime
import json

from mpi4py import MPI
import numpy as np
import gym

from src import logger
from src.imagine.rl.ddpg import DDPG
from src.utils.util import find_save_path, set_global_seeds
from src.utils.nlp_tools import OneHotEncoder, analyze_descr, Vocab
from src.imagine.rl.her import make_sample_her_transitions as make_sample_her_transitions_biased
from src.imagine.reward_function.classifier_reward_function_lstm_pretrained import RewardFunctionLSTMPretrained
from src.imagine.reward_function.classifier_reward_function_lstm_learned import RewardFunctionLSTM
from src.imagine.reward_function.oracle_reward_function_playground import OracleRewardFunction as OraclePlayground
from src.playground_env.descriptions import train_descriptions as train_descriptions_env
from src.playground_env.descriptions import test_descriptions as test_descriptions_env
from src.playground_env.descriptions import extra_descriptions as extra_descriptions_env
from src.imagine.goal_generator.descriptions import get_descriptions
from src.playground_env.env_params import N_OBJECTS_IN_SCENE, ENV_ID
from src.imagine.language_model import LanguageModelLSTM

HOME = os.environ['HOME']
if 'flowers' in HOME:
    USE_LOCAL_CONFIG = True
else:
    USE_LOCAL_CONFIG = False
REPO_PATH = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-3]) + '/'

# This is the default configuration dict. Main parameters are overridden by parameters defined in train.py
DEFAULT_CONFIG = dict(experiment_params=dict(trial_id=0,
                                             logdir=None,
                                             n_epochs=500,  # 1 epoch is n_cpus * n_cycles * rollout_batch_size episodes
                                             n_cycles=50,  # number of cycles per epoch
                                             rollout_batch_size=2,  # rollouts per mpi thread (per cycle)
                                             n_batches=50,  # number of updates per cycle
                                             n_cpus=6,  # number of mpi threads
                                             seed=np.random.randint(int(1e6)),
                                             policy_save_interval=10,  # the interval with which policy pickles are saved (in epochs)
                                             save_obs=False,  # whether to save observation to build a dataset
                                             method_test='robin'  # test all instructions one after the other
                                             ),
                      conditions=dict(env_name='PlaygroundNavigation-v1',
                                      env_id=ENV_ID,
                                      policy_architecture='modular_attention',  # 'flat_concat', 'flat_attention', 'modular_attention'
                                      imagination_method='CGH',
                                      policy_encoding='lstm',  # policy encoding
                                      reward_checkpoint='',  # filepath of reward function checkpoint
                                      goal_invention='from_epoch_70',  # format: from_epoch_XX
                                      p_imagined=0.5,  # probability to use imagined goals in imagination phases
                                      reward_function='learned_lstm',  # or 'pretrained'
                                      feedback_strategy='exhaustive',  # 'exhaustive' or  'one_pos_one_neg'
                                      p_social_partner_availability=1,  # probability for SP to be present
                                      rl_positive_ratio=.5,  # ratio of positive examples in policy batches
                                      reward_positive_ratio=.2  # ratio of positive examples per goal in reward function batches
                                      ),
                      # learning parameters of DDPG, from OpenAI Baselines HER implementation
                      learning_params=dict(algo='ddpg',  # choice of underlying learning algorithm (ddpg or td3 supported so far)
                                           normalize_obs=False,  # whether observation are normalized by running stats
                                           norm_eps=0.01,  # epsilon used for observation normalization
                                           norm_clip=5,  # normalized observations are cropped to this values
                                           clip_return=1,  # whether or not returns should be clipped
                                           layers=1,  # number of hidden layers in the critic/actor networks
                                           hidden=256,  # number of neurons in each hidden layer
                                           network_class='src.imagine.rl.actor_critic:ActorCritic',
                                           Q_lr=0.001,  # critic learning rate
                                           pi_lr=0.001,  # actor learning rate
                                           buffer_size=int(1e6),  # for experience replay
                                           batch_size=256,  # for rl updates
                                           polyak=0.95,  # polyak averaging coefficient
                                           action_l2=1.0,  # quadratic penalty on actions (before rescaling by max_u)
                                           clip_obs=200.,
                                           rollout_batch_size=2,  # rollouts per mpi thread
                                           scope='ddpg',  # can be tweaked for testing
                                           random_eps=0.3,  # percentage of time a random action is taken
                                           noise_eps=0.2,  # std for gaussian noise on the actions
                                           test_with_polyak=False
                                           ),
                      reward_function=dict(batch_size=512,
                                           max_n_epoch=100,
                                           early_stopping='f1',
                                           n_batch=200,  # number of batches per epoch
                                           freq_update=2,  # train the reward function every x RL training epochs
                                           n_objs=N_OBJECTS_IN_SCENE,
                                           learning_rate=0.001,
                                           ff_size=100,  # size of hidden layer in reward function
                                           num_hidden_lstm=100  # number of hidden states in language model lstm
                                           )
                      )

if USE_LOCAL_CONFIG:
    # debug configuration, runs quickly but does not learn
    DEFAULT_CONFIG['reward_function']['max_n_epoch'] = 5
    DEFAULT_CONFIG['experiment_params']['n_cycles'] = 10
    DEFAULT_CONFIG['experiment_params']['policy_save_interval'] = 1


def configure_everything(rank, seed, num_cpu, env, trial_id, n_epochs, reward_function, policy_encoding,
                         feedback_strategy, policy_architecture, goal_invention, reward_checkpoint,
                         rl_positive_ratio, p_partner_availability, imagination_method, git_commit=''):

    # Seed everything
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # goal invention one epoch later than stated
    epoch = int(goal_invention.split('_')[-1]) + 1
    goal_invention = 'from_epoch_{}'.format(epoch)

    # Prepare params.
    params = DEFAULT_CONFIG
    train_descriptions, test_descriptions, all_descriptions = get_descriptions(ENV_ID)
    assert sorted(train_descriptions) == sorted(train_descriptions_env)
    assert sorted(test_descriptions) == sorted(test_descriptions_env)
    params.update(date_time=str(datetime.datetime.now()),
                  train_descriptions=train_descriptions,
                  test_descriptions=test_descriptions,
                  extra_descriptions=extra_descriptions_env,
                  all_descriptions=all_descriptions,
                  git_commit=git_commit
                  )

    # Configure logging
    if rank == 0:
        logdir = find_save_path('../../data/expe/' + env + "/", trial_id)
        logger.configure(dir=logdir)
        os.makedirs(logdir + 'tmp/', exist_ok=True)
        os.makedirs(logdir + 'reward_checkpoints/', exist_ok=True)
        os.makedirs(logdir + 'policy_checkpoints/', exist_ok=True)
        os.makedirs(logdir + 'goal_info/', exist_ok=True)
        if params['experiment_params']['save_obs']:
            os.makedirs(logdir + 'save_obs/', exist_ok=True)
    else:
        logdir = None
    logdir = MPI.COMM_WORLD.bcast(logdir, root=0)

    # Update conditions parameters from arguments or variables defined in train.py
    params['conditions'].update(env_name=env,
                                policy_architecture=policy_architecture,
                                reward_function=reward_function,
                                goal_invention=goal_invention,
                                imagination_method=imagination_method,
                                feedback_strategy=feedback_strategy,
                                rl_positive_ratio=rl_positive_ratio,
                                reward_checkpoint=reward_checkpoint,
                                policy_encoding=policy_encoding,
                                p_social_partner_availability=p_partner_availability
                                )

    # checks
    if params['conditions']['policy_architecture'] in ['modular_attention', 'attention']:
        error_msg =  'You need an lstm policy encoding and reward is you use {}'.format(params['conditions']['policy_architecture'])
        assert params['conditions']['policy_encoding'] == 'lstm', error_msg
        assert params['conditions']['reward_function'] in ['pretrained', 'learned_lstm'], error_msg
    elif params['conditions']['reward_function'] == 'oracle':
        error_msg =  'You cannot use an lstm policy encoding if you use an oracle reward'
        assert params['conditions']['policy_encoding'] != 'lstm', error_msg
        error_msg =  'You can only use a flat_concat policy architecture if you use an oracle reward'
        assert params['conditions']['policy_architecture'] == 'flat_concat', error_msg


    # Update experiment parameters from arguments or variables defined in train.py
    params['experiment_params'].update(n_epochs=n_epochs,
                                       trial_id=trial_id,
                                       logdir=logdir,
                                       seed=seed,
                                       n_cpus=num_cpu,
                                       n_test_rollouts=len(params['train_descriptions']),
                                       )
    params['reward_function'].update(reward_positive_ratio=params['conditions']['reward_positive_ratio'])
    # Define social partner params
    params['social_partner_params'] = dict(feedback_strategy=feedback_strategy,
                                           p_availability=p_partner_availability)

    # Env generating function
    def make_env():
        return gym.make(params['conditions']['env_name'])

    # Get info from environment and configure dimensions dict
    tmp_env = make_env()
    tmp_env.reset()
    params['learning_params']['T'] = tmp_env._max_episode_steps
    params['learning_params']['gamma'] = 1. - 1. / params['learning_params']['T']

    if params['conditions']['policy_encoding'] == 'lstm':
        dim_encoding = params['reward_function']['num_hidden_lstm']
    else:
        raise NotImplementedError

    inds_objs = tmp_env.unwrapped.inds_objs  # indices of object in state
    for i in range(len(inds_objs)):
        inds_objs[i] = inds_objs[i].tolist()
    dims = dict(obs=tmp_env.observation_space.shape[0],
                g_encoding=dim_encoding,
                g_id=1,
                acts=tmp_env.action_space.shape[0],
                g_str=None,
                nb_obj=tmp_env.unwrapped.nb_obj,
                inds_objs=inds_objs)
    params['dims'] = dims

    # configure learning params and interactions
    if params['learning_params']['algo'] == 'ddpg':
        params['learning_params']['network_class'] += 'DDPG'
    else:
        raise NotImplementedError

    params['training_rollout_params'] = dict(exploit=False,
                                             use_target_net=False,
                                             compute_Q=False,
                                             eval_bool=False,
                                             )
    params['evaluation_rollout_params'] = dict(exploit=True,
                                               use_target_net=params['learning_params']['test_with_polyak'],
                                               compute_Q=True,
                                               eval_bool=True
                                               )

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        params['training_rollout_params'][name] = params['learning_params'][name]
        params['evaluation_rollout_params'][name] = params['learning_params'][name]
    params['evaluation_rollout_params']['rollout_batch_size'] = 1

    params['repo_path'] = REPO_PATH
    params['lstm_reward_checkpoint_path'] = REPO_PATH + '/src/data/lstm_checkpoints/{}'.format(params['conditions']['reward_checkpoint'])
    params['or_params_path'] = dict()
    for n_obj in [3]:
        params['or_params_path'][n_obj] = REPO_PATH + '/src/data/or_function/or_params_{}objs.pk'.format(n_obj)

    # Save parameter dict
    if rank == 0:
        with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
            json.dump(params, f)
        for key in sorted(params.keys()):
            logger.info('{}: {}'.format(key, params[key]))

    params['make_env'] = make_env

    return params, rank_seed

def get_one_hot_encoder():
    _, max_seq_length, word_set = analyze_descr(get_descriptions(ENV_ID)[2])
    vocab = Vocab(word_set)
    one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
    return one_hot_encoder


def get_reward_function(goal_sampler, params):
    rew_func = params['conditions']['reward_function']
    if rew_func == 'oracle':
        if 'PlaygroundNavigation' in params['conditions']['env_name']:
            return OraclePlayground(goal_sampler, params, convert_from_discovery_ids=True)
        else:
            raise NotImplementedError
    elif rew_func == 'pretrained':
        return RewardFunctionLSTMPretrained(goal_sampler, params)
    elif rew_func == 'learned_lstm':
        return RewardFunctionLSTM(goal_sampler, params)
    else:
        raise NotImplementedError


def get_oracle_reward_function(goal_sampler, params):
    if 'PlaygroundNavigation' in params['conditions']['env_name']:
        return OraclePlayground(goal_sampler, params)
    else:
        raise NotImplementedError


def get_language_models(params):
    if 'lstm' in params['conditions']['reward_function'] or params['conditions']['reward_function'] == 'pretrained':
        reward_language_model = LanguageModelLSTM(params=params)
    elif params['conditions']['reward_function'] == 'oracle':
        reward_language_model = None
    else:
        raise NotImplementedError

    if params['conditions']['policy_encoding'] == 'lstm':
        if reward_language_model is not None:
            policy_language_model = reward_language_model
        else:
            policy_language_model = LanguageModelLSTM(params=params)
    else:
        raise NotImplementedError

    return policy_language_model, reward_language_model


def configure_her(params, reward_function, goal_sampler):
    def reward_fun(state, goal):  # vectorized
        return reward_function.predict(state, goal)

    # Prepare configuration for HER.
    params['conditions']['rl_positive_ratio'] = float(params['conditions']['rl_positive_ratio'])
    sample_her_transitions = make_sample_her_transitions_biased(goal_sampler=goal_sampler,
                                                                goal_invention=params['conditions']['goal_invention'],
                                                                p_imagined=params['conditions']['p_imagined'],
                                                                rl_positive_ratio=params['conditions']['rl_positive_ratio'],
                                                                reward_fun=reward_fun)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_learning_algo(reward_function, goal_sampler, params, reuse=False, use_mpi=True):
    sample_her_transitions = configure_her(params, reward_function, goal_sampler)
    params['dims'].update(action_max=1)
    # Learning agent
    params['learning_params'].update(input_dims=params['dims'].copy(),  # agent takes an input observations
                                     clip_pos_returns=True,  # clip positive returns
                                     clip_return=(1. / (1. - params['learning_params']['gamma'])) if
                                     params['learning_params']['clip_return'] else np.inf,  # max abs of return
                                     goal_sampler=goal_sampler,
                                     policy_architecture=params['conditions']['policy_architecture'],
                                     reward_function=reward_function,
                                     dims=params['dims'].copy(),
                                     cuda=False,
                                     logdir=params['experiment_params']['logdir'],
                                     clip_range=5,
                                     lr_actor=0.001,
                                     lr_critic=0.001,
                                     alpha=0.2,
                                     )

    if params['learning_params']['algo'] == 'ddpg':
        policy = DDPG(params['learning_params'], sample_her_transitions)
    else:
        raise NotImplementedError
    return policy
