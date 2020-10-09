from collections import OrderedDict, deque
import time
import pickle
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from src import logger
from src.utils.util import import_function, flatten_grads
from src.imagine.rl.replay_buffer import ReplayBuffer
from src.utils.mpi_util import MpiAdam


class DDPG(object):
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, action_l2, clip_obs, scope, T,
                 clip_pos_returns, clip_return,
                 sample_transitions, gamma, policy_architecture=None, reward_function=None, goal_sampler=None, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'imagine.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each or_module
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        self.input_dims = input_dims
        self.buffer_size = buffer_size
        self.hidden = hidden
        self.layers = layers
        self.network_class = network_class
        self.policy_architecture = policy_architecture
        self.polyak = polyak
        self.batch_size = batch_size
        self.Q_lr = Q_lr
        self.pi_lr = pi_lr
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.action_l2 = action_l2
        self.clip_obs = clip_obs
        self.scope = scope
        self.T = T
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.sample_transitions = sample_transitions
        self.gamma = gamma
        self.reuse = reuse
        self.policy_perturb_scale = 0.01

        if self.clip_return is None:
            self.clip_return = np.inf
        self.create_actor_critic = import_function(self.network_class)

        self.dimo = self.input_dims['obs']
        self.dimg = self.input_dims['g_encoding']
        self.dimu = self.input_dims['acts']
        self.dimgid = self.input_dims['g_id']
        self.inds_objs = self.input_dims['inds_objs']

        # Prepare staging area for feeding data to the model.
        buffer_shapes = dict(acts=(self.T, self.input_dims['acts']),
                             obs=(self.T + 1, self.input_dims['obs']),
                             g_id=(self.input_dims['g_id'],),
                             g_encoding=(1, self.input_dims['g_encoding']),
                             )
        self.stage_shapes = OrderedDict(acts=(None, self.input_dims['acts']),
                                   obs=(None, self.input_dims['obs']),
                                   g_id=(None,),
                                   g_encoding=(None, self.input_dims['g_encoding']),
                                   r=(None,),
                                   obs_2=(None, self.input_dims['obs']),
                                   )


        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                                          shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer
        self.buffer = ReplayBuffer(buffer_shapes,
                                   self.buffer_size,
                                   self.T,
                                   self.sample_transitions,
                                   goal_sampler,
                                   reward_function)
        self.replay_proba = None
        self.memory_replay_ratio_positive_rewards = deque(maxlen=50)
        self.memory_replay_ratio_positive_per_goal = deque(maxlen=50)


    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, self.dimu))

    def _preprocess_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, g)
        policy = self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {policy.o_tf: o.reshape(-1, self.dimo),
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
                }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_attention(self, g):
        feed = {self.main.g_tf: g}
        return self.sess.run(self.main.attention, feed_dict=feed)

    def get_critic_attention(self, g):
        feed = {self.main.g_tf: g}
        return self.sess.run(self.main.critic_attention, feed_dict=feed)

    def store_episode(self, episode_batch, goals_reached_id, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch, goals_reached_id)

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([self.Q_loss_tf,
                                                                  self.main.Q_pi_tf,
                                                                  self.Q_grad_tf,
                                                                  self.pi_grad_tf
                                                                  ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    
    def sample_batch(self, epoch):
        out = self.buffer.sample(self.batch_size, epoch)
        transitions, replay_ratio_positive_rewards, self.replay_proba, replay_ratio_positive_per_goal, time_dict = out
        self.memory_replay_ratio_positive_rewards.append(replay_ratio_positive_rewards)
        self.memory_replay_ratio_positive_per_goal.append(replay_ratio_positive_per_goal)
        
        o, o_2, g = transitions['obs'], transitions['obs_2'], transitions['g_encoding']
        transitions['obs'], transitions['g_encoding'] = self._preprocess_og(o, g)
        transitions['obs_2'], _ = self._preprocess_og(o_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch, time_dict

    def get_replay_ratio_positive_reward_stat(self):
        return np.mean(self.memory_replay_ratio_positive_rewards)

    def get_replay_ratio_positive_per_goal_stat(self):
        return np.nanmean(self.memory_replay_ratio_positive_per_goal, axis=0)

    def stage_batch(self, epoch, batch=None):
        if batch is None:
            batch, time_dict = self.sample_batch(epoch)
        assert len(self.buffer_ph_tf) == len(batch)
        init = time.time()
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))
        time_dict['time_run_batch'] = time.time() - init
        return time_dict

    def train(self, epoch, stage=True):
        timee = time.time()
        if stage:
            time_dict= self.stage_batch(epoch)
        times_training = dict(time_batch=time.time() - timee)
        timee = time.time()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        times_training.update(time_update=time.time() - timee, **time_dict)
        return critic_loss, actor_loss, times_training

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def save_params(self, path):
        variables = self._vars('')
        names = [v.name for v in variables]
        values = [v.eval() for v in variables]
        to_save = dict(zip(names, values))
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)

    def load_params(self, path):
        variables = self._vars('')
        with open(path, 'rb') as f:
            params = pickle.load(f)
        assign_ops = []
        for v in variables:
            assign_ops.append(v.assign(params[v.name]))
        self.sess.run(assign_ops)

    def _create_network(self, reuse=False):
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("Creating a DDPG agent with action space %d..." % (self.dimu))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['obs'] = batch_tf['obs_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def reset_policy(self):
        tf.variables_initializer(self._global_vars('')).run()

    def logs(self, prefix=''):
        logs = []

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions','params',
                             'stage_shapes', 'create_actor_critic', 'reward_function', 'rollout']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name and 'rollout' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name and 'rollout' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
