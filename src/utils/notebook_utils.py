import argparse
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pygame
import numpy as np

sys.path.append('../../../')

import src.imagine.experiment.config as config

from src.imagine.goal_sampler import GoalSampler


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


def generate_animation_reward_module(env, goal_str, reward_language_model, reward_function, policy):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plots = []
    rewards1 = []
    rewards2 = []
    rewards3 = []
    ## Rollout of a policy
    env.reset()
    initial_o = env.unwrapped.reset_with_goal(goal_str)
    o = initial_o.copy()
    goal_encoding = reward_language_model.encode(goal_str)
    input_goal = torch.tensor(goal_encoding).float().view(1, len(goal_encoding))
    objects = [obj.object_descr['colors'] + ' ' + obj.object_descr['types'] for obj in env.objects]

    for t in range(30):
        action = policy.get_actions(o, goal_encoding)
        o, _, _, _ = env.step(action)
        input_o = torch.tensor(o).float().view(1, len(o))
        reward_per_object = reward_function.reward_function.compute_logits_before_or(input_o, input_goal)
        r1, r2, r3 = (elem.detach()[0][0].item() for elem in reward_per_object)
        rewards1.append(r1)
        rewards2.append(r2)
        rewards3.append(r3)
        env.render(close=True)
        obs = pygame.surfarray.array3d(env.viewer).transpose([1, 0, 2])
        im = ax1.imshow(obs, animated=True)
        line1, = ax2.plot(rewards1, color='blue', label='R for ' + objects[0])
        line2, = ax2.plot(rewards2, color='red', label='R for ' + objects[1])
        line3, = ax2.plot(rewards3, color='green', label='R for ' + objects[2])
        if t == 0:
            ax2.legend()
        plots.append([im, line1, line2, line3])

    ani = animation.ArtistAnimation(fig, plots, interval=400, blit=False,
                                    repeat=False)
    return ani


def generate_animation_policy_module(env, goal_str, reward_language_model, policy):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plots = []
    z_list1 = []
    z_list2 = []
    z_list3 = []
    ## Rollout of a policy
    env.reset()
    initial_o = env.unwrapped.reset_with_goal(goal_str)
    o = initial_o.copy()
    goal_encoding = reward_language_model.encode(goal_str)
    input_goal = torch.tensor(goal_encoding).float().view(1, len(goal_encoding))
    objects = [obj.object_descr['colors'] + ' ' + obj.object_descr['types'] for obj in env.objects]

    for t in range(30):
        action = policy.get_actions(o, goal_encoding)
        o, _, _, _ = env.step(action)
        input_o = torch.tensor(o).float().view(1, len(o))
        norm_per_object = policy.actor_network.get_norm_per_object(input_o, input_goal)
        z1, z2, z3 = (elem.detach().item() for elem in norm_per_object)
        z_list1.append(z1)
        z_list2.append(z2)
        z_list3.append(z3)
        env.render(close=True)
        obs = pygame.surfarray.array3d(env.viewer).transpose([1, 0, 2])
        im = ax1.imshow(obs, animated=True)
        line1, = ax2.plot(z_list1, color='blue', label='|z| for ' + objects[0])
        line2, = ax2.plot(z_list2, color='red', label='|z| for ' + objects[1])
        line3, = ax2.plot(z_list3, color='green', label='|z| for ' + objects[2])
        if t == 0:
            ax2.legend()
        plots.append([im, line1, line2, line3])

    ani = animation.ArtistAnimation(fig, plots, interval=400, blit=False,
                                    repeat=False)
    return ani


def plot_attention_vector(attention_vector, goal_str, params):
    body_feat = ['Body X', 'Body Y', 'Body gripper']
    obj_things = list(params['env_params']['name_attributes'])[:-5]
    obj_feat = ['obj X', 'obj Y', 'obj size', 'obj R', 'obj G', 'obj B', 'obj gripper']
    delta_body_feat = ['Δ ' + elem for elem in body_feat]
    delta_obj_things = ['Δ ' + elem for elem in obj_things]
    delta_obj_feat = ['Δ ' + elem for elem in obj_feat]

    state_description = body_feat + delta_body_feat + obj_things + obj_feat + delta_obj_things + delta_obj_feat

    def show_values(pc, state_description, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), state_description):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, value, ha="center", va="center", color=color, **kw)

    fig = plt.figure(figsize=(6, 12))
    attention_vector = attention_vector.reshape([len(state_description), 1])
    c = plt.pcolor(attention_vector)
    plt.title(goal_str)
    plt.colorbar()
    show_values(c, state_description)
    fig.tight_layout()


def plot_tsne(X_embedded, descr, code, params):
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111)
    sc = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], picker=5)
    normal_colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                     [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
                     [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                     [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]
    colors_colors = ((171 / 255, 16 / 255, 16 / 255), (0, 81 / 255, 159 / 255), (10 / 255, 145 / 255, 10 / 255))

    def color_code(x):
        if x == 'colors':
            firsts = [None] * 4
            colors = ('red', 'blue', 'green')
            groups = [None] * len(descr)
            for i_d, d in enumerate(descr):
                found = False
                for i in range(len(colors)):
                    if colors[i] in d:
                        if firsts[i] == None:
                            firsts[i] = i_d
                        groups[i_d] = i
                        found = True
                        break
                if not found:
                    if firsts[3] == None:
                        firsts[3] = i_d

            colors = []
            for i in range(len(descr)):
                if groups[i] is None:
                    colors.append('k')
                else:
                    colors.append(colors_colors[groups[i]])
            sc.set_facecolor(colors)
            legend = ('red', 'blue', 'green', 'any')
            plots = [plt.scatter(X_embedded[firsts[i], 0],
                                 X_embedded[firsts[i], 1],
                                 c=(list(colors_colors) + ['k'])[i]) for i in range(4)]
            plt.legend(plots, legend,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.15),
                       ncol=4,
                       fancybox=True,
                       shadow=True,
                       prop={'size': 10},
                       markerscale=2., )
        elif x == 'predicates':
            firsts = [None] * 3
            predicates = ('Grasp', 'Go', 'Grow')
            groups = [None] * len(descr)
            for i_d, d in enumerate(descr):
                for i in range(len(predicates)):
                    if predicates[i] in d:
                        if firsts[i] == None:
                            firsts[i] = i_d
                        groups[i_d] = i
                        break
            colors = []
            for i in range(len(descr)):
                if groups[i] is None:
                    colors.append('k')
                else:
                    colors.append(colors_colors[groups[i]])
            sc.set_facecolor(colors)
            legend = ('Grasp', 'Go', 'Grow')
            plots = [plt.scatter(X_embedded[firsts[i], 0],
                                 X_embedded[firsts[i], 1],
                                 c=colors_colors[i]) for i in range(3)]
            plt.legend(plots, legend,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.15),
                       ncol=3,
                       fancybox=True,
                       shadow=True,
                       prop={'size': 10},
                       markerscale=2., )
        elif x == 'categories':
            cats = sorted(list(params['env_params']['categories'].keys()))
            cats.remove('living_thing')
            categories = params['env_params']['categories']
            firsts = [None] * (len(categories) + 1)

            groups = [None] * len(descr)
            for i_d, d in enumerate(descr):
                found = False
                for i, cat in enumerate(cats):
                    is_cat = any([w_type in d for w_type in categories[cat]])
                    if is_cat:
                        if firsts[i] == None:
                            firsts[i] = i_d
                        groups[i_d] = i
                        found = True
                        break
                if not found:
                    if firsts[-1] == None:
                        firsts[-1] = i_d

            colors = []
            for i in range(len(descr)):
                if groups[i] is None:
                    colors.append('k')
                else:
                    colors.append(normal_colors[groups[i]])
            sc.set_facecolor(colors)
            legend = cats + ['None']
            plots = [plt.scatter(X_embedded[firsts[i], 0],
                                 X_embedded[firsts[i], 1],
                                 c=normal_colors[i]) for i in range(len(cats) + 1)]
            plt.legend(plots, legend,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.15),
                       ncol=3,
                       fancybox=True,
                       shadow=True,
                       prop={'size': 10},
                       markerscale=2., )
        else:
            raise NotImplementedError

    color_code(code)
