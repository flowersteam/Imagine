
import pickle
import os
import json
import sys
sys.path.append('../../')
from src.utils.util import set_global_seeds
import src.imagine.experiment.config as config
from src.imagine.interaction import RolloutWorker
from src.imagine.goal_sampler import GoalSampler
from src.playground_env.reward_function import get_reward_from_state
from src.playground_env.env_params import thing_colors, plants
from src.utils.util import get_stat_func
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import gym
import argparse
font = {'size'   : 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
                  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]


PATH = 'path_to_folder_with_trial_ids_folders/'

RUN = True
PLOT = True
FREQ = 10
RENDER = 0
N_REPET = 1
LINE = 'mean'
ERR = 'std'


plants.remove('flower')
types_words = [['Grasp red tree', 'Grasp blue door', 'Grasp green dog', 'Grow green dog'],
               ['flower'],
               ['Grasp {} animal'.format(d) for d in thing_colors + ['any']],
               ['Grasp {} fly'.format(d) for d in thing_colors + ['any']],
               ['Grow {} {}'.format(c, p) for c in thing_colors + ['any'] for p in plants + ['plant', 'living_thing']]
               ]
types_words[4].remove('Grow red tree')
type_legends = ['Type {}'.format(i) for i in range(1, len(types_words) + 1)]
n_types = len(type_legends)


def run_generalization_study(path, freq=10):
    first = True

    for t_id, trial in enumerate(os.listdir(path)):
        print(trial)
        t_init = time.time()
        trial_folder = path + '/' + trial + '/'
        policy_folder = trial_folder + 'policy_checkpoints/'
        params_file = trial_folder + 'params.json'

        data = pd.read_csv(os.path.join(trial_folder, 'progress.csv'))
        all_epochs = data['epoch']
        all_episodes = data['episode']
        epochs = []
        episodes = []
        for epoch, episode in zip(all_epochs, all_episodes):
            if epoch % freq == 0:
                epochs.append(epoch)
                episodes.append(int(episode))

        # Load params
        with open(params_file) as json_file:
            params = json.load(json_file)
        seed = params['experiment_params']['seed']
        set_global_seeds(seed)

        goal_invention = int(params['conditions']['goal_invention'].split('_')[-1])
        test_descriptions = params['test_descriptions']

        rank = 0
        if first:
            if not RENDER:
                env = 'PlaygroundNavigation-v1'
            else:
                env = 'PlaygroundNavigationRender-v1'
            params, rank_seed = config.configure_everything(rank=rank,
                                                            seed=seed,
                                                            num_cpu=params['experiment_params']['n_cpus'],
                                                            env=env,
                                                            trial_id=0,
                                                            n_epochs=10,
                                                            reward_function=params['conditions']['reward_function'],
                                                            policy_encoding=params['conditions']['policy_encoding'],
                                                            bias_buffer=params['conditions']['bias_buffer'],
                                                            feedback_strategy=params['conditions']['feedback_strategy'],
                                                            policy_architecture=params['conditions']['policy_architecture'],
                                                            goal_invention=params['conditions']['goal_invention'],
                                                            reward_checkpoint=params['conditions']['reward_checkpoint'],
                                                            rl_positive_ratio=params['conditions']['rl_positive_ratio'],
                                                            p_partner_availability=params['conditions']['p_social_partner_availability'],
                                                            git_commit='')

            policy_language_model, reward_language_model = config.get_language_models(params)
            onehot_encoder = config.get_one_hot_encoder()
            goal_sampler = GoalSampler(policy_language_model=policy_language_model,
                                       reward_language_model=reward_language_model,
                                       goal_dim=policy_language_model.goal_dim,
                                       one_hot_encoder=onehot_encoder,
                                       **params['goal_sampler'],
                                       params=params)


            reward_function = config.get_reward_function(goal_sampler, params)
        else:
            def make_env():
                return gym.make(params['conditions']['env_name'])

            params['make_env'] = make_env
        loaded = False
        success_rates = np.zeros([len(test_descriptions), len(epochs)])
        if params['conditions']['reward_function'] == 'pretrained':
            reward_function.load_params(trial_folder + 'params_reward')
        if not loaded:
            # Load policy.
            t_init = time.time()

            for ind_ep, epoch in enumerate(epochs):
                print(time.time() - t_init)
                t_init = time.time()

                print('\n\n\t\t EPOCH', epoch)
                if first:
                    first = False
                    reuse = False
                else:
                    reuse = True

                if params['conditions']['reward_function'] == 'learned_lstm':
                    reward_function.restore_from_checkpoint(trial_folder + 'reward_checkpoints/reward_func_checkpoint_{}'.format(epoch))

                policy_language_model.set_reward_function(reward_function)
                if reward_language_model is not None:
                    reward_language_model.set_reward_function(reward_function)

                goal_sampler.update_discovered_goals(params['all_descriptions'], episode_count=0, epoch=0)

                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    with open(policy_folder + 'policy_{}.pkl'.format(epoch), 'rb') as f:
                        policy = pickle.load(f)

                evaluation_worker = RolloutWorker(make_env=params['make_env'],
                                                  policy=policy,
                                                  reward_function=reward_function,
                                                  params=params,
                                                  render=RENDER,
                                                  **params['evaluation_rollout_params'])
                evaluation_worker.seed(seed)

                # Run evaluation.
                evaluation_worker.clear_history()
                successes_per_descr = np.zeros([len(test_descriptions)])
                for ind_inst, instruction in enumerate(test_descriptions):
                    # instruction = 'Grasp any fly'
                    success_instruction = []
                    goal_str = [instruction]
                    goal_encoding = [policy_language_model.encode(goal_str[0])]
                    goal_id = [0]
                    for i in range(N_REPET):
                        ep = evaluation_worker.generate_rollouts(exploit=True,
                                                                 imagined=False,
                                                                 goals_str=goal_str,
                                                                 goals_encodings=goal_encoding,
                                                                 goals_ids=goal_id)
                        success = get_reward_from_state(state=ep[0]['obs'][-1], goal=instruction)
                        success_instruction.append(success)
                    success_rate_inst = np.mean(success_instruction)
                    successes_per_descr[ind_inst] = success_rate_inst
                    print('\t Success rate {}: {}'.format(goal_str[0], success_rate_inst))
                    success_rates[ind_inst, ind_ep] = success_rate_inst
                np.savetxt(trial_folder + 'generalization_success_rates.txt', success_rates)

def plot_generalization(path, freq):

    for trial in os.listdir(path):
        print(trial)
        t_init = time.time()
        trial_folder = path + '/' + trial + '/'
        policy_folder = trial_folder + 'policy_checkpoints/'
        params_file = trial_folder + 'params.json'

        data = pd.read_csv(os.path.join(trial_folder, 'progress.csv'))
        all_epochs = data['epoch']
        all_episodes = data['episode']
        epochs = []
        episodes = []
        for epoch, episode in zip(all_epochs, all_episodes):
            if epoch % freq == 0:
                epochs.append(epoch)
                episodes.append(int(episode))

        # Load params
        with open(params_file) as json_file:
            params = json.load(json_file)
        seed = params['experiment_params']['seed']
        set_global_seeds(seed)

        goal_invention = int(params['conditions']['goal_invention'].split('_')[-1])
        test_descriptions = params['test_descriptions']

        success_rates = np.loadtxt(path + '/' + trial + '/generalization_success_rates.txt')

        line, err_min, err_max = get_stat_func(LINE, ERR)
        first = False
        # plot
        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(6)
        ax.spines['right'].set_linewidth(6)
        ax.spines['bottom'].set_linewidth(6)
        ax.spines['left'].set_linewidth(6)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        plt.plot(np.array(episodes) / 1000, line(success_rates), linewidth=10)
        plt.fill_between(np.array(episodes) / 1000, err_min(success_rates), err_max(success_rates), alpha=0.2)
        if goal_invention < 100:
            plt.vlines(goal_invention * 0.6, ymin=0, ymax=1, linestyles='--', color='k', linewidth=5)
        lab = plt.xlabel('Episodes (x$10^3$)')
        plt.ylim([-0.01, 1.01])
        plt.yticks([0.25, 0.50, 0.75, 1])
        lab2 = plt.ylabel('Average success rate')
        plt.savefig(os.path.join(trial_folder, 'generalization_test_set_policy.pdf'), bbox_extra_artists=(lab, lab2), bbox_inches='tight',
                    dpi=50)  # add leg

        # plot per group
        inds_per_types = []
        descr_per_type = []
        for i_type, type in enumerate(types_words):
            inds_per_types.append([])
            descr_per_type.append([])
            for i_d, descr in enumerate(test_descriptions):
                for type_w in type:
                    if type_w in descr:
                        inds_per_types[-1].append(i_d)
                        descr_per_type[-1].append(descr)
            inds_per_types[-1] = np.array(inds_per_types[-1])
        for i in range(len(type_legends)):
            print('Type {}:'.format(i + 1), descr_per_type[i])

        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(6)
        ax.spines['right'].set_linewidth(6)
        ax.spines['bottom'].set_linewidth(6)
        ax.spines['left'].set_linewidth(6)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(len(types_words)):
            to_plot = success_rates[np.array(inds_per_types[i]), :]
            plt.plot(np.array(episodes) / 1000 , line(to_plot), linewidth=8, c=colors[i])
            plt.fill_between(np.array(episodes) / 1000 , err_min(to_plot), err_max(to_plot), color=colors[i], alpha=0.2)
        if goal_invention < 100:
            plt.vlines(goal_invention * 0.6, ymin=0, ymax=1, linestyles='--', color='k', linewidth=5)
        leg = plt.legend(type_legends, frameon=False)
        lab = plt.xlabel('Episodes (x$10^3$)')
        plt.ylim([-0.01, 1.01])
        plt.yticks([0.25, 0.50, 0.75, 1])
        lab2 = plt.ylabel('Average success rate')
        plt.savefig(os.path.join(trial_folder, 'generalization_test_set_policy_per_type.pdf'), bbox_extra_artists=(lab, lab2), bbox_inches='tight',
                    dpi=50)  # add leg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--path', type=str, default=PATH)
    add('--plot', default=PLOT, type=lambda x: (str(x).lower() == 'true'))
    add('--run', default=RUN, type=lambda x: (str(x).lower() == 'true'))
    kwargs = vars(parser.parse_args())
    if kwargs['run']:
        run_generalization_study(kwargs['path'], FREQ)
    if kwargs['plot']:
        plot_generalization(kwargs['path'], FREQ)