import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
font = {'size'   : 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
                  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]


folder_path = 'path_to_folder_with_trial_ids_folders/'


track_time = True

def plot_all(path, trial):
    print('Ploting trial', trial)
    plt.close('all')

    # extract params from json
    with open(os.path.join(path, 'params.json')) as json_file:
        params = json.load(json_file)

    instructions = params['train_descriptions']

    nb_instr = len(instructions)
    n_cycles = params['experiment_params']['n_cycles']
    rollout_batch_size = params['experiment_params']['rollout_batch_size']
    n_cpu = params['experiment_params']['n_cpus']

    # extract notebooks
    data = pd.read_csv(os.path.join(path, 'progress.csv'))

    n_points = data['eval/success_goal_0'].shape[0]
    episodes = data['episode']

    n_epochs = len(episodes)
    n_eps = n_cpu * rollout_batch_size * n_cycles
    episodes = np.arange(n_eps, n_epochs * n_eps + 1, n_eps)
    episodes = episodes / 1000

    task_success_rates = np.zeros([n_points, nb_instr])
    goals_reached = np.zeros([n_points, nb_instr])
    for i in range(nb_instr):
        task_success_rates[:, i] = data['eval/success_goal_' + str(i)]
    zero_success_rates = task_success_rates.copy()
    for i in range(zero_success_rates.shape[0]):
        for j in range(zero_success_rates.shape[1]):
            if np.isnan(zero_success_rates[i, j]):
                zero_success_rates[i, j] = 0
    np.savetxt(path + 'sr_train_set.txt', zero_success_rates.transpose())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    # plot success_rate

    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    p = plt.plot(episodes, zero_success_rates.mean(axis=1), linewidth=10)  # , c=colors[i])
    # leg = plt.legend(['task ' + str(i) for i in range(nb_instr)], frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Average success rate')
    plt.savefig(os.path.join(path, 'plot_av_success_rate.png'), bbox_extra_artists=(lab, lab2), bbox_inches='tight',
                dpi=50)  # add leg
    plt.close('all')

    if track_time:
        computation_time = data['epoch_duration (s)']
        time_batch = data['time_batch']
        time_epoch = data['time_epoch']
        time_train = data['time_train']
        time_update = data['time_update']
        time_replay = data['time_replay']
        time_reward_func_replay = data['time_reward_func_replay']
        time_reward_func_update = data['time_reward_func_update']

        # check training timings
        time_stuff = [time_epoch, time_train - time_update, time_update, time_batch, time_replay, time_reward_func_replay, time_reward_func_update]
        legends = ['time_epoch', 'time_train', 'time_update', 'time_batch', 'time_replay', 'time_reward_func_replay', 'time_reward_func_update']

        discoveries = np.zeros([nb_instr])
        discoveries.fill(np.nan)
        for i in range(nb_instr):
            ind_zero = np.argwhere(goals_reached[:, i] == 0)
            if ind_zero.size == 0:
                discoveries[i] = 0
            else:
                discoveries[i] = ind_zero[-1][0]

        np.savetxt(os.path.join(path, 'discoveries.txt'), discoveries)

        # plot computation time per epoch
        fig = plt.figure(figsize=(22, 15), frameon=False)
        plt.plot(episodes, computation_time, linewidth=3)
        for d in discoveries:
            plt.axvline(x=episodes[int(d)])
        plt.savefig(os.path.join(path, 'time.png'), bbox_extra_artists=(lab, lab2), bbox_inches='tight',
                    dpi=50)  # add leg
        plt.close('all')

        fig = plt.figure(figsize=(22, 15), frameon=False)
        for i in range(len(time_stuff)):
            plt.plot(episodes, time_stuff[i], linewidth=3)
        plt.legend(legends)
        for d in discoveries:
            plt.axvline(x=episodes[int(d)])
        plt.savefig(os.path.join(path, 'time2.png'), bbox_extra_artists=(lab, lab2), bbox_inches='tight',
                    dpi=50)  # add leg
    plt.close('all')

    # Extract exploration scores
    first = 600
    step = 600
    last = int(np.array(episodes)[-1]) * 1000
    steps = np.arange(first, last + 1, step)
    n_steps = steps.size
    var_obj_pos = np.zeros([n_steps])
    count_reward_extra_set = np.zeros([n_steps])
    count_reward_train_set = np.zeros([n_steps])
    count_reward_test_set = np.zeros([n_steps])
    dist_per_obj = np.zeros([n_steps])
    exploration_score_all = np.zeros([n_steps])
    exploration_score_test = np.zeros([n_steps])
    exploration_score_train = np.zeros([n_steps])
    exploration_score_extra = np.zeros([n_steps])
    var_states = np.zeros([n_steps])
    # counter_special = np.zeros([n_steps])
    counter_rew_train_test = np.zeros([n_steps])

    for k, ep in enumerate(steps):
        with open(path + 'goal_info/info_' + str(ep) + '.pk', 'rb') as f:
            data = pickle.load(f)
        metrics = data['exploration_metrics']

        # track number of reward and exploration score
        explo_score_train = 0
        explo_score_all = 0
        explo_score_test = 0
        explo_score_extra = 0
        prev_counters = dict()
        rarities = []
        for d in metrics['counter_since_begininng'].keys():
            prev_counters[d] = metrics['counter_since_begininng'][d] - metrics['rewards_last_state'][d]
            rarities.append(1 / (1 + prev_counters[d]))
            assert prev_counters[d] >= 0

        for d in params['train_descriptions']:
            explo_score_train += metrics['rewards_last_state'][d] * (1 / (prev_counters[d] + 1))

        for d in params['test_descriptions']:
            explo_score_test += metrics['rewards_last_state'][d] * (1 / (prev_counters[d] + 1))

        for d in params['extra_descriptions']:

            explo_score_extra += metrics['rewards_last_state'][d] * (1 / (prev_counters[d] + 1))

        for d in params['train_descriptions'] + params['test_descriptions'] + params['extra_descriptions']:
            explo_score_all += metrics['rewards_last_state'][d] * (1 / (prev_counters[d] + 1))

        # for d in specialial[t_i, k] += metrics['rewards_last_state'][d]

        explo_score_train /= np.mean(rarities)
        explo_score_test /= np.mean(rarities)
        explo_score_extra /= np.mean(rarities)
        explo_score_all /= np.mean(rarities)

        exploration_score_train[k] = explo_score_train
        exploration_score_test[k] = explo_score_test
        exploration_score_extra[k] = explo_score_extra
        exploration_score_all[k] = explo_score_all

        dist_per_obj[k] = metrics['dist_per_obj']
        count_reward_test_set[k] = metrics['count_reward_test_set']
        count_reward_train_set[k] = metrics['count_reward_train_set']
        count_reward_extra_set[k] = metrics['count_reward_extra_set']
        counter_rew_train_test[k] = metrics['count_reward_test_set'] + metrics['count_reward_train_set']
        var_obj_pos[k] = metrics['var_obj_pos']
        var_states[k] = metrics['var_states']

    exploration_metrics = dict(var_obj_pos=var_obj_pos,
                               count_reward_extra_set=count_reward_extra_set,
                               count_reward_train_set=count_reward_train_set,
                               count_reward_test_set=count_reward_test_set,
                               dist_per_obj=dist_per_obj,
                               exploration_score_all=exploration_score_all,
                               exploration_score_test=exploration_score_test,
                               exploration_score_train=exploration_score_train,
                               exploration_score_extra=exploration_score_extra,
                               var_states=var_states,
                               counter_rew_train_test=counter_rew_train_test
                               )
    with open(path + 'exploration_metrics.pk', 'wb') as f:
        pickle.dump(exploration_metrics, f)



if __name__=="__main__":
    for trial in os.listdir(folder_path  + '/'):
        path = folder_path + '/' + trial + '/'
        plot_all(path, trial)
