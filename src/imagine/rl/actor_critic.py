import tensorflow as tf
from src.utils.util import store_args, nn
import numpy as np

class ActorCriticDDPG:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, inds_objs, hidden, layers,policy_architecture='modular_attention', **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (imagine.her.Normalizer): normalizer for observations
            g_stats (imagine.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.layers = layers
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.hidden = hidden
        self.o_tf = inputs_tf['obs']
        self.g_tf = inputs_tf['g_encoding']
        self.u_tf = inputs_tf['acts']

        half_o = int(self.o_tf.shape[1]) // 2
        n_objs = len(inds_objs)
        dim_obj = 2 * len(inds_objs[0])
        dim_body = inds_objs[0][0] * 2

        o = self.o_tf
        g = self.g_tf


        if policy_architecture == 'modular_attention':

            # here we use the same net to map [obj + body] x attention to a latent space
            # Then we sum these representations to use them as input of the policy / critic
            # this implements a deep set

            with tf.variable_scope('pi'):
                # cast embedding to attention in [0, 1]
                self.attention = tf.nn.sigmoid(nn(g, [dim_body + dim_obj], name="attention"))

                obs_body = tf.concat(axis=1, values=[o[:, :inds_objs[0][0]],
                                                     o[:, half_o: half_o + inds_objs[0][0]]])
                input_objs = []

                # implement deepset
                for i in range(n_objs):
                    obs_obj = tf.concat(axis=1, values=[o[:, inds_objs[i][0]: inds_objs[i][-1] + 1],
                                             o[:, inds_objs[i][0] + half_o: inds_objs[i][-1] + 1 + half_o]])
                    body_obj_input = tf.concat(axis=1, values=[obs_body, obs_obj])

                    # apply attention with Hadamard product
                    deepset_input = tf.multiply(body_obj_input, self.attention)

                    # use a same representer for all_obj
                    input_obj = tf.nn.relu(nn(deepset_input, [self.hidden] + [n_objs * (dim_obj + dim_body)], name='obj', reuse=i>0))
                    input_objs.append(input_obj)
                # sum the latent representations
                input_pi = tf.add_n(input_objs)
                # final network to compute actions
                self.pi_tf = tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu], name='pi'))
            with tf.variable_scope('Q'):
                # build attention from embedding
                self.critic_attention = tf.nn.sigmoid(nn(g, [dim_body + dim_obj + self.dimu], name="attention"))

                obs_body = tf.concat(axis=1, values=[o[:, :inds_objs[0][0]],
                                                     o[:, half_o: half_o + inds_objs[0][0]]])
                input_objs = []
                # implement deepset
                for i in range(n_objs):
                    obs_obj = tf.concat(axis=1, values=[o[:, inds_objs[i][0]: inds_objs[i][-1] + 1],
                                                        o[:, inds_objs[i][0] + half_o: inds_objs[i][-1] + 1 + half_o]])
                    body_obj_act_input = tf.concat(axis=1, values=[obs_body, obs_obj, self.pi_tf])

                    # apply attention
                    deepset_input = tf.multiply(body_obj_act_input, self.critic_attention)

                    # use a same representer for all_obj
                    input_obj = tf.nn.relu(nn(deepset_input, [self.hidden] + [n_objs * (dim_obj + dim_body + self.dimu)], name='obj', reuse=i > 0))
                    input_objs.append(input_obj)
                input_Q = tf.add_n(input_objs)
                self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1], name='critic')


                # for critic training
                # build attention from embedding
                self.critic_attention = tf.nn.sigmoid(nn(g, [dim_body + dim_obj + self.dimu], name="attention", reuse=True))

                obs_body = tf.concat(axis=1, values=[o[:, :inds_objs[0][0]],
                                                     o[:, half_o: half_o + inds_objs[0][0]]])
                input_objs2 = []
                # implement deepset
                for i in range(n_objs):
                    obs_obj = tf.concat(axis=1, values=[o[:, inds_objs[i][0]: inds_objs[i][-1] + 1],
                                                        o[:, inds_objs[i][0] + half_o: inds_objs[i][-1] + 1 + half_o]])
                    body_obj_act_input2 = tf.concat(axis=1, values=[obs_body, obs_obj, self.u_tf])

                    # apply attention
                    deepset_input2 = tf.multiply(body_obj_act_input2, self.critic_attention)

                    # use a same representer for all_obj
                    input_obj2 = tf.nn.relu(nn(deepset_input2, [self.hidden] + [n_objs * (dim_obj + dim_body + self.dimu)], name='obj', reuse=True))
                    input_objs2.append(input_obj2)
                input_Q2 = tf.add_n(input_objs2)
                self._input_Q = input_Q2  # exposed for tests
                self.Q_tf = nn(input_Q2, [self.hidden] * self.layers + [1], name='critic', reuse=True)

        elif policy_architecture == 'modular_concat':

            # here we use the same multi layer non linear net to map [obj + body] x attention to a latent space
            # Then we sum these representations to use them as input of the policy / critic
            # this implements a deep set

            with tf.variable_scope('pi'):
                # build attention from embedding

                obs_body = tf.concat(axis=1, values=[o[:, :inds_objs[0][0]],
                                                     o[:, half_o: half_o + inds_objs[0][0]]])
                input_objs = []
                # implement deepset
                for i in range(n_objs):
                    obs_obj = tf.concat(axis=1, values=[o[:, inds_objs[i][0]: inds_objs[i][-1] + 1],
                                             o[:, inds_objs[i][0] + half_o: inds_objs[i][-1] + 1 + half_o]])
                    body_obj_input = tf.concat(axis=1, values=[obs_body, obs_obj])

                    # apply attention
                    deepset_input = tf.concat(axis=1, values=[body_obj_input, g])

                    # use a same representer for all_obj
                    input_obj = tf.nn.relu(nn(deepset_input, [self.hidden] + [n_objs * (dim_obj + dim_body)], name='obj', reuse=i>0))
                    input_objs.append(input_obj)
                input_pi = tf.add_n(input_objs)
                self.pi_tf = tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu], name='pi'))
            with tf.variable_scope('Q'):
                # build attention from embedding

                obs_body = tf.concat(axis=1, values=[o[:, :inds_objs[0][0]],
                                                     o[:, half_o: half_o + inds_objs[0][0]]])
                input_objs = []
                # implement deepset
                for i in range(n_objs):
                    obs_obj = tf.concat(axis=1, values=[o[:, inds_objs[i][0]: inds_objs[i][-1] + 1],
                                                        o[:, inds_objs[i][0] + half_o: inds_objs[i][-1] + 1 + half_o]])
                    body_obj_act_input = tf.concat(axis=1, values=[obs_body, obs_obj, self.pi_tf])

                    # apply attention
                    deepset_input = tf.concat(axis=1, values=[body_obj_act_input, g])

                    # use a same representer for all_obj
                    input_obj = tf.nn.relu(nn(deepset_input, [self.hidden] + [n_objs * (dim_obj + dim_body + self.dimu)], name='obj', reuse=i > 0))
                    input_objs.append(input_obj)
                input_Q = tf.add_n(input_objs)
                self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1], name='critic')


                # for critic training
                # build attention from embedding

                obs_body = tf.concat(axis=1, values=[o[:, :inds_objs[0][0]],
                                                     o[:, half_o: half_o + inds_objs[0][0]]])
                input_objs2 = []
                # implement deepset
                for i in range(n_objs):
                    obs_obj = tf.concat(axis=1, values=[o[:, inds_objs[i][0]: inds_objs[i][-1] + 1],
                                                        o[:, inds_objs[i][0] + half_o: inds_objs[i][-1] + 1 + half_o]])
                    body_obj_act_input2 = tf.concat(axis=1, values=[obs_body, obs_obj, self.u_tf])

                    # apply attention
                    deepset_input2 = tf.concat(axis=1, values=[body_obj_act_input2, g])

                    # use a same representer for all_obj
                    input_obj2 = tf.nn.relu(nn(deepset_input2, [self.hidden] + [n_objs * (dim_obj + dim_body + self.dimu)], name='obj', reuse=True))
                    input_objs2.append(input_obj2)
                input_Q2 = tf.add_n(input_objs2)
                self._input_Q = input_Q2  # exposed for tests
                self.Q_tf = nn(input_Q2, [self.hidden] * self.layers + [1], name='critic', reuse=True)
        elif policy_architecture == 'flat_concat':
            # Networks.
            with tf.variable_scope('pi'):
                input_pi = tf.concat(axis=1, values=[o, g])  # for actor
                self.pi_tf = tf.tanh(nn(input_pi, [self.hidden] * 2 + [self.dimu]))
            with tf.variable_scope('Q'):
                input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf])
                self.Q_pi_tf = nn(input_Q, [self.hidden] * 2 + [1])
                # for critic training
                input_Q = tf.concat(axis=1, values=[o, g, self.u_tf])
                self._input_Q = input_Q  # exposed for tests
                self.Q_tf = nn(input_Q, [self.hidden] * 2 + [1], reuse=True)

        elif policy_architecture == 'flat_attention':
            # Networks.
            with tf.variable_scope('pi'):
                # build attention from embedding
                attention = tf.nn.sigmoid(nn(g, [o.shape[1]], name="attention"))
                # apply attention
                input_pi = tf.multiply(o, attention)
                self.pi_tf = tf.tanh(nn(input_pi, [self.hidden] * 2 + [self.dimu], name='pi'))
            with tf.variable_scope('Q'):
                # build attention from embedding
                attention = tf.nn.sigmoid(nn(g, [o.shape[1] + self.dimu], name="attention"))
                obs_act_input = tf.concat(axis=1, values=[o, self.pi_tf])
                # apply attention
                input_Q = tf.multiply(obs_act_input, attention)
                self.Q_pi_tf = nn(input_Q, [self.hidden] * 2 + [1], name='critic')

                # for critic training
                # build attention from embedding
                attention = tf.nn.sigmoid(nn(g, [o.shape[1] + self.dimu], name="attention", reuse=True))
                obs_act_input = tf.concat(axis=1, values=[o, self.u_tf])
                # apply attention
                input_Q2 = tf.multiply(obs_act_input, attention)
                self.Q_tf = nn(input_Q2, [self.hidden] * 2 + [1], name='critic', reuse=True)
        else:
            raise NotImplementedError

