import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib
import json

font = {'size'   : 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

folder = '/home/flowers/Desktop/Scratch/Curious-nlp/src/results/PlaygroundNavigation-v1/'
# trial_ids = [160, 161, 162, 163]
# trial_ids = [184, 189, 188, 190]
# legs = ['goal sampler, goal invention 50',
#         'goal sampler, no goal invention',
#         'goal sampler, goal invention 10',
#         'no goal sampler, no goal invention']
trial_ids = [201, 203, 202]
legs = ['goal invention 80',
        'goal invention 500',
        'goal invention 10']

# trial_ids = [160, 165, 179, 181]
# legs = ['old gs cost', 'new gs rar2, prob1 cost', 'new gs rar2 prob2, cost', 'new gs rar3 prob2, cost']

# trial_ids = [167, 189, 182]
# legs = ['new gs rar2, prob1 f1', 'new gs rar2 prob2, f1', 'new gs rar3 prob2, f1']

to_plot_str = ['var_obj_pos', 'count_reward_extra_set', 'count_reward_train_set', 'count_reward_test_set',
           'dist_per_obj', 'exploration_score_train', 'exploration_score_test', 'exploration_score_all',
              'exploration_score_extra', 'var_states', 'counter_special', 'counter_rew_train_test']
first = 600
step = 600
last = 0

# find last
for t_id in trial_ids:
    path = os.path.join(folder, str(t_id) + '/')
    data_csv = pd.read_csv(os.path.join(path, 'progress.csv'))
    episodes = data_csv['episode']
    t_id_last = int(np.array(episodes)[-1])
    if t_id_last > last:
        last = t_id_last

with open(folder + str(trial_ids[0]) + '/params.json', 'r') as f:
    params = json.load(f)

special_goals = ['Feed dark unicorn', 'Feed dark dragon', 'Feed dark phoenix']
steps = np.arange(first, last + 1, step)
n_steps = steps.size
var_obj_pos = np.zeros([len(trial_ids), n_steps])
var_obj_pos.fill(np.nan)
count_reward_extra_set = np.zeros([len(trial_ids), n_steps])
count_reward_extra_set.fill(np.nan)
count_reward_train_set = np.zeros([len(trial_ids), n_steps])
count_reward_train_set.fill(np.nan)
count_reward_test_set = np.zeros([len(trial_ids), n_steps])
count_reward_test_set.fill(np.nan)
dist_per_obj = np.zeros([len(trial_ids), n_steps])
dist_per_obj.fill(np.nan)
exploration_score_all = np.zeros([len(trial_ids), n_steps])
exploration_score_all.fill(np.nan)
exploration_score_test = np.zeros([len(trial_ids), n_steps])
exploration_score_test.fill(np.nan)
exploration_score_train = np.zeros([len(trial_ids), n_steps])
exploration_score_train.fill(np.nan)
exploration_score_extra = np.zeros([len(trial_ids), n_steps])
exploration_score_extra.fill(np.nan)
var_states = np.zeros([len(trial_ids), n_steps])
var_states.fill(np.nan)
counter_special = np.zeros([len(trial_ids), n_steps])
counter_rew_train_test = np.zeros([len(trial_ids), n_steps])
counter_rew_train_test.fill(np.nan)

to_plot = [var_obj_pos, count_reward_extra_set, count_reward_train_set, count_reward_test_set,
           dist_per_obj, exploration_score_train, exploration_score_test, exploration_score_all,
              exploration_score_extra, var_states, counter_special, counter_rew_train_test]
for t_i, trial_id in enumerate(trial_ids):
    path = os.path.join(folder, str(trial_id) + '/')
    data_csv = pd.read_csv(os.path.join(path, 'progress.csv'))
    episodes = data_csv['episode']
    last = int(np.array(episodes)[-1])
    t_id_steps = np.arange(first, last + 1, step)

    for k, ep in enumerate(t_id_steps):
        with open(path + 'goal_info/info_' + str(ep) + '.pk', 'rb') as f:
            data = pickle.load(f)
        metrics = data['exploration_metrics']

        if k > 0:
            counter_special[t_i, k] = counter_special[t_i, k - 1]
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

        exploration_score_train[t_i, k] = explo_score_train
        exploration_score_test[t_i, k] = explo_score_test
        exploration_score_extra[t_i, k] = explo_score_extra
        exploration_score_all[t_i, k] = explo_score_all

        dist_per_obj[t_i, k] = metrics['dist_per_obj']
        count_reward_test_set[t_i, k] = metrics['count_reward_test_set']
        count_reward_train_set[t_i, k] = metrics['count_reward_train_set']
        count_reward_extra_set[t_i, k] = metrics['count_reward_extra_set']
        counter_rew_train_test[t_i, k] = metrics['count_reward_test_set'] + metrics['count_reward_train_set']
        var_obj_pos[t_i, k] = metrics['var_obj_pos']
        var_states[t_i, k] = metrics['var_states']
    stop = 1

for i in range(len(to_plot_str)):
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    toplot = to_plot[i].copy()
    for j in range(2, toplot.shape[1] - 2):
        toplot[:, j] = np.nanmean(to_plot[i][:, j - 2: j + 2], axis=1)
    p = plt.plot(steps, toplot.transpose(), linewidth=10)  # , c=colors[i])
    # leg = plt.legend(['task ' + str(i) for i in range(nb_instr)], frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    # plt.ylim([-0.01, 1.01])
    # plt.yticks([0.25, 0.50, 0.75, 1])
    plt.legend(legs)
    plt.title(to_plot_str[i])
    plt.savefig(os.path.join(folder, to_plot_str[i] + '.png'), bbox_extra_artists=(lab,), bbox_inches='tight', dpi=50)  # add leg




