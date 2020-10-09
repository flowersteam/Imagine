import torch
from collections import deque
import time
import pickle
import numpy as np
from src.imagine.rl.mpi_utils.mpi_utils import sync_networks, sync_grads
from src.imagine.rl.replay_buffer import ReplayBuffer
from src.imagine.rl.actor_critic import Actor, Critic
from src.imagine.rl.mpi_utils.normalizer import Normalizer


class DDPG:
    """
    Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
    Taken from https://github.com/TianhongDai/hindsight-experience-replay

    sample_transitions (function) function that samples from the replay buffer
    Args forwarded in params (dict)
        dims (dict of ints): dimensions for the observation (obs), the goal (g_encoding), and the
            actions (acts) and action max (action_max)
        buffer_size (int): number of transitions that are stored in the replay buffer
        hidden (int): number of units in the hidden layers
        layers (int): number of hidden layers
        polyak (float): coefficient for Polyak-averaging of the target network
        batch_size (int): batch size for training
        lr_critic (float): learning rate for the Q (critic) network
        lr_actor (float): learning rate for the pi (actor) network
        norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
        norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
        action_l2 (float): coefficient for L2 penalty on the actions
        clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
        T (int): the time horizon for rollouts
        clip_pos_returns (boolean): whether or not positive returns should be clipped
        gamma (float): gamma used for Q learning updates
    """

    def __init__(self, params, sample_transitions):
        params = params.copy()
        params['make_env'] = None

        self.dims = params['dims']
        self.cuda = params['cuda']
        self.logdir = params['logdir']
        self.clip_range = params['clip_range']
        # self.positive_ratio = params['positive_ratio'] defined in her sample transition
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.buffer_size = params['buffer_size']
        # self.env_name = params['env_name']
        self.random_eps = params['random_eps']
        self.noise_eps = params['noise_eps']
        self.polyak = params['polyak']
        self.clip_obs = params['clip_obs']
        self.action_l2 = params['action_l2']
        self.normalize = params['normalize_obs']
        # self.batch_size = params['batch_size'] defined in her
        self.gamma = params['gamma']
        self.model_path = params['logdir'] + '/policies/'
        self.T = params['T']
        self.reward_function = params['reward_function']
        self.sample_transitions = sample_transitions
        self.batch_size = params['batch_size']

        hidden = params['hidden']
        layers = params['layers']

        # create the replay buffer
        buffer_shapes = dict(acts=(self.T, self.dims['acts']),
                             obs=(self.T + 1, self.dims['obs']),
                             g_id=(self.dims['g_id'],),
                             g_encoding=(1, self.dims['g_encoding']),
                             )
        self.buffer = ReplayBuffer(buffer_shapes,
                                   self.buffer_size,
                                   self.T,
                                   self.sample_transitions,
                                   params['goal_sampler'],
                                   self.reward_function)

        self.replay_proba = None
        self.memory_replay_ratio_positive_rewards = deque(maxlen=50)
        self.memory_replay_ratio_positive_per_goal = deque(maxlen=50)

        # create the network
        self.actor_network = Actor(params['dims'], layers, hidden)
        self.critic_network = Critic(params['dims'], layers, hidden)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = Actor(params['dims'], layers, hidden)
        self.critic_target_network = Critic(params['dims'], layers, hidden)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.lr_critic)

        # create the normalizer
        self.o_norm = Normalizer(size=params['dims']['obs'], default_clip_range=self.clip_range)
        self.g_norm = Normalizer(size=params['dims']['g_encoding'], default_clip_range=self.clip_range)

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'o_norm', 'g_norm', 'env', 'sample_transitions', 'params',
                             'goal_sampler',
                             'stage_shapes', 'create_actor_critic', 'reward_function', 'rollout']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size

        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None


    def _preproc_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        if self.normalize:
            obs_norm = self.o_norm.normalize(obs)
            g_norm = self.g_norm.normalize(g)
            # concatenate the stuffs
            inputs = np.concatenate([obs_norm, g_norm])
        else:
            inputs = np.concatenate([obs, g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.cuda:
            inputs = inputs.cuda()
        return inputs

    def _preproc_inputs_no_concat(self, obs, g):
        if self.normalize:
            obs_norm = self.o_norm.normalize(obs)
            g_norm = self.g_norm.normalize(g)
            # concatenate the stuffs
        else:
            obs_norm = obs
            g_norm = g
        obs_norm = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        g_norm = torch.tensor(g_norm, dtype=torch.float32).unsqueeze(0)
        if self.cuda:
            obs_norm = obs_norm.cuda()
            g_norm = g_norm.cuda()
        return obs_norm, g_norm

    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, self.dims['acts']))

    def get_actions(self, obs, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        with torch.no_grad():
            obs, g = self._preproc_inputs_no_concat(obs, g)
            action = self.actor_network(obs, g)
            if compute_Q:
                q_value = self.critic_network(obs, g, action)
            u = action.cpu().numpy()
            noise = noise_eps * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (
                    self._random_action(u.shape[0]) - u)  # eps-greedy
            if u.shape[0] == 1:
                u = u[0]
            u = u.copy()
            action = u
            if compute_Q:
                return action, q_value
            else:
                return action

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    def soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)

    def update(self, epoch):
        times_training = self._update_network(epoch)
        return times_training

    def save_model(self, path):
        torch.save(
            [self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
            path)

    def load_params(self, path):
        model = torch.load(path)
        self.o_norm.mean= model[0]
        self.o_norm.std = model[1]
        self.g_norm.mean = model[2]
        self.g_norm.std = model[3]
        self.actor_network.load_state_dict(model[-1])


    def store_episode(self, episodes, goals_reached_ids):
        self.buffer.store_episode(episodes, goals_reached_ids)

    def sample_batch(self, epoch):
        out = self.buffer.sample(self.batch_size, epoch)
        transitions, replay_ratio_positive_rewards, self.replay_proba, replay_ratio_positive_per_goal, time_dict = out
        self.memory_replay_ratio_positive_rewards.append(replay_ratio_positive_rewards)
        self.memory_replay_ratio_positive_per_goal.append(replay_ratio_positive_per_goal)

        # pre-process the observation and goal
        o, o_2, g = transitions['obs'], transitions['obs_2'], transitions['g_encoding']
        transitions['obs'], transitions['g_encoding'] = self._preproc_og(o, g)
        transitions['obs_2'], _ = self._preproc_og(o_2, g)
        return transitions, time_dict

    # update the network
    def _update_network(self, epoch):

        timee = time.time()
        # sample the episodes
        transitions, time_dict = self.sample_batch(epoch)
        # start to do the update

        if self.normalize:
            obs_norm = torch.tensor(self.o_norm.normalize(transitions['obs']), dtype=torch.float32)
            g_norm = torch.tensor(self.g_norm.normalize(transitions['g_encoding']), dtype=torch.float32)
            obs_next_norm = torch.tensor(self.o_norm.normalize(transitions['obs_2']), dtype=torch.float32)
            g_next_norm = torch.tensor(self.g_norm.normalize(transitions['g_encoding']), dtype=torch.float32)
        else:
            obs_norm = torch.tensor(transitions['obs'], dtype=torch.float32)
            g_norm = torch.tensor(transitions['g_encoding'], dtype=torch.float32)
            obs_next_norm = torch.tensor(transitions['obs_2'], dtype=torch.float32)
            g_next_norm = torch.tensor(transitions['g_encoding'], dtype=torch.float32)

        actions_tensor = torch.tensor(transitions['acts'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.cuda:
            obs_norm = obs_norm.cuda()
            g_norm = g_norm.cuda()
            obs_next_norm = obs_next_norm.cuda()
            g_next_norm = g_next_norm.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function

        times_training = dict(time_batch=time.time() - timee)
        timee = time.time()
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(obs_next_norm, g_next_norm)
            q_next_value = self.critic_target_network(obs_next_norm, g_next_norm, actions_next)
            q_next_value = q_next_value.squeeze().detach()
            target_q_value = r_tensor + self.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(obs_norm, g_norm, actions_tensor).squeeze()
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(obs_norm, g_norm)
        actor_loss = -self.critic_network(obs_norm, g_norm, actions_real).mean()
        actor_loss += self.action_l2 * (actions_real / self.dims['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        times_training.update(time_update=time.time() - timee, **time_dict)
        return times_training

    def get_replay_ratio_positive_reward_stat(self):
        return np.mean(self.memory_replay_ratio_positive_rewards)

    def get_replay_ratio_positive_per_goal_stat(self):
        return np.nanmean(self.memory_replay_ratio_positive_per_goal, axis=0)

    def logs(self, prefix=''):
        logs = []

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
