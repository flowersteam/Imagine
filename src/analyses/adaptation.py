
import pickle
import os
import json
import sys
sys.path.append('../../')
from src.utils.util import set_global_seeds
import src.imagine.experiment.config as config
from src.imagine.interaction import RolloutWorker
from src.imagine.goal_sampler import GoalSampler
from src.playground_env.reward_function import get_reward_from_state, water_on_furniture, food_on_furniture
from src.playground_env.env_params import thing_colors, plants
from src.utils.util import get_stat_func
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import gym
font = {'size'   : 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
                  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]


PATH = '/media/flowers/3C3C66F13C66A59C/ICML2020/results/new_res/ImaginedGoals/big_plant02/'
FREQ = 10
RENDER = 0
N_REPET = 30
LINE = 'mean'
ERR = 'std'


def plot_generalization(path, freq=10):
    first = True
    trial_folder = path
    for trial in os.listdir(path):
        print(trial)
        # if os.path.exists(path + '/' + trial + '/adaptation_success_rates_food.txt'):

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
        env_id = params['conditions']['env_id']
        if 'plant' not in env_id:
            test_plants = plants.copy() + ['plant', 'living_thing']
            test_plants.remove('flower')
            test_descriptions = ['Grow {} {}'.format(c, p) for c in thing_colors + ['any'] for p in test_plants]
        else:
            if 'big' in env_id:
                test_plants = ['algae', 'bonsai', 'tree', 'bush', 'plant', 'living_thing']
            else:
                test_plants = ['tree', 'bush', 'plant', 'living_thing']
            test_descriptions = ['Grow {} {}'.format(c, p) for c in thing_colors + ['any'] for p in test_plants]


        first_epoch = True

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
                                                            curriculum_replay_target=params['conditions']['curriculum_replay_target'],
                                                            curriculum_target=params['conditions']['curriculum_target'],
                                                            policy_encoding=params['conditions']['policy_encoding'],
                                                            bias_buffer=params['conditions']['bias_buffer'],
                                                            feedback_strategy=params['conditions']['feedback_strategy'],
                                                            goal_sampling_policy=params['conditions']['goal_sampling_policy'],
                                                            policy_architecture=params['conditions']['policy_architecture'],
                                                            goal_invention=params['conditions']['goal_invention'],
                                                            reward_checkpoint=params['conditions']['reward_checkpoint'],
                                                            rl_positive_ratio=params['conditions']['rl_positive_ratio'],
                                                            p_partner_availability=params['conditions']['p_social_partner_availability'],
                                                            power_rarity=2,
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





        # Load policy.
        success_rates = np.zeros([len(test_descriptions), len(epochs), 2])
        for ind_ep, epoch in enumerate(epochs):
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
            successes_per_descr = np.zeros([len(test_descriptions), 2])
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
                    for t in range(ep[0]['obs'].shape[0]):
                        metric_food = food_on_furniture(ep[0]['obs'][t], goal_str[0])
                        if metric_food:
                            # print('\n\n Touched food')
                            break
                    for t in range(ep[0]['obs'].shape[0]):
                        metric_water = water_on_furniture(ep[0]['obs'][t], goal_str[0])
                        if metric_water:
                            # print('\n \n Touched water')
                            break
                    success_instruction.append([metric_food, metric_water])
                success_instruction = np.array(success_instruction)
                success_rate_inst = np.mean(success_instruction, axis=0)
                successes_per_descr[ind_inst] = success_rate_inst
                print('\t Success rate {}: food {}, water {}'.format(goal_str[0], success_rate_inst[0], success_rate_inst[1]))
                success_rates[ind_inst, ind_ep, :] = success_rate_inst
            np.savetxt(trial_folder + 'adaptation_success_rates_water.txt', success_rates[:, :, 1])
            np.savetxt(trial_folder + 'adaptation_success_rates_food.txt', success_rates[:, :, 0])

        # success_rates = np.zeros([len(test_descriptions), len(epochs), 2])
        # success_rates[:, :, 0] = np.loadtxt(trial_folder + 'adaptation_success_rates_food.txt')
        # success_rates[:, :, 1] = np.loadtxt(trial_folder + 'adaptation_success_rates_water.txt')

        line, err_min, err_max = get_stat_func(LINE, ERR)
        # plot
        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(6)
        ax.spines['right'].set_linewidth(6)
        ax.spines['bottom'].set_linewidth(6)
        ax.spines['left'].set_linewidth(6)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(2):
            plt.plot(np.array(episodes) / 1000, line(success_rates)[:, i], linewidth=10, color=colors[i])
            plt.fill_between(np.array(episodes) / 1000, err_min(success_rates)[:, i], err_max(success_rates)[:, i], color=colors[i], alpha=0.2)
        # plt.vlines(goal_invention * 0.6, ymin=0, ymax=1, linestyles='--', color='k', linewidth=5)
        leg = plt.legend(['food', 'water'], frameon=False)
        lab = plt.xlabel('Episodes (x$10^3$)')
        plt.ylim([-0.01, 1.01])
        plt.yticks([0.25, 0.50, 0.75, 1])
        lab2 = plt.ylabel('Average success rate')
        plt.savefig(os.path.join(trial_folder, 'adaptation_success_rates.pdf'), bbox_extra_artists=(lab, lab2, leg), bbox_inches='tight',
                    dpi=50)  # add leg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--path', type=str, default=PATH)
    kwargs = vars(parser.parse_args())
    plot_generalization(kwargs['path'], FREQ)
