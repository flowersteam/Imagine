import tensorflow as tf
from tensorflow.contrib import rnn
from src.utils.util import nn
import pickle

class TFRewardFunction(object):
    '''
    Reward function parent class that has no tensorflow graph but that return all the desired metrics
    for training and testing
    '''

    def __init__(self, state_size, voc_size, sequence_length):
        self.state_size = state_size
        self.voc_size = voc_size
        self.sequence_length = sequence_length

        self.pred = None
        self.cost = None
        self.optimizer = None
        self.acc = None
        self.prec = None
        self.recall = None
        self.h_lstm = None

    def get_summary(self):
        return tf.summary.merge_all()

    def get_cost(self):
        return self.cost

    def get_optimizer(self):
        return self.optimizer

    def get_accuracy(self):
        return self.acc

    def get_precision(self):
        return self.prec

    def get_recall(self):
        return self.recall

    def get_instruction_embedding(self):
        return self.h_lstm

    def get_pred(self):
        return self.pred


def RNN(x, sequence_length, num_hidden_lstm):
    '''
    x: input, input shape must be (batch_size, sequence_length, input_size)
    '''
    x = tf.unstack(x, sequence_length, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden_lstm, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return outputs[-1]


def compute_confusion(pred, Y):
    tp = tf.math.logical_and(tf.equal(pred, tf.constant(1, "float")),
                             tf.equal(Y, tf.constant(1, "float")))
    fp = tf.math.logical_and(tf.equal(pred, tf.constant(1, "float")),
                             tf.equal(Y, tf.constant(0, "float")))

    tn = tf.math.logical_and(tf.equal(pred, tf.constant(0, "float")),
                             tf.equal(Y, tf.constant(0, "float")))
    fn = tf.math.logical_and(tf.equal(pred, tf.constant(0, "float")),
                             tf.equal(Y, tf.constant(1, "float")))
    return tp, fp, tn, fn


class OrFunction:
    '''
    Pretrained or function init from path to parameters file
    '''

    # Create model
    def __init__(self, path_to_dict):
        with open(path_to_dict, 'rb') as f:
            self.data = pickle.load(f)
        self.layers_sizes = self.data['layers']
        self.nb_layers = len(self.layers_sizes)
        self.n_obj = self.layers_sizes[0][0]
        self.layers = []
        for i in range(self.nb_layers):
            self.layers.append(self.create_layer(layer_id=i,
                                                 size=self.layers_sizes[i],
                                                 weight=self.data['fc{}_weight'.format(i + 1)],
                                                 bias=self.data['fc{}_bias'.format(i + 1)]))
        self.activations = []
        for i in range(self.nb_layers):
            if self.data['activations'][i] == 'relu':
                self.activations.append(tf.nn.relu)
            elif self.data['activations'][i] == 'sigmoid':
                self.activations.append(tf.nn.sigmoid)
            elif self.data['activations'][i] == 'tanh':
                self.activations.append(tf.nn.tanh)
            else:
                raise NotImplementedError

    def forward(self, x):
        for i in range(self.nb_layers):
            x = self.layers[i](x)
            x = self.activations[i](x)
        x = 0.98 * x + 0.01
        return x

    def create_layer(self, layer_id, size, weight, bias):
        weight_init = weight.transpose()
        bias_init = bias
        W = tf.get_variable('or_l{}_w'.format(layer_id), dtype=tf.float32, initializer=weight_init, trainable=False)
        b = tf.get_variable('or_l{}_b'.format(layer_id), dtype=tf.float32, initializer=bias_init, trainable=False)

        def fc(x):
            return tf.nn.bias_add(tf.matmul(x, W), b)

        return fc


class RewardFunctionConcat(TFRewardFunction):
    '''
    The flat concatenation (FC) model of the reward function
    '''
    def __init__(self, state_size, voc_size, sequence_length, batch_size, learning_rate, ff_size, num_hidden_lstm):
        super().__init__(state_size, voc_size, sequence_length)
        self.learning_rate = learning_rate
        self.beta = 0.00
        self.batch_size = batch_size
        self.ff_size = ff_size
        self.num_hidden_lstm = num_hidden_lstm  # self.sub_state_size

        # creating the graph
        with tf.variable_scope('reward'):
            with tf.variable_scope('inputs'):
                self.S = tf.placeholder("float", [None, self.state_size], name='state')
                self.I = tf.placeholder("float", [None, self.sequence_length, self.voc_size], name='instruction')
                self.Y = tf.placeholder("float", [None, 1], name='reward')
                self.precomputed_h_lstm = tf.placeholder("float", [None, self.num_hidden_lstm],
                                                         name='precomputed_hlstm')

            with tf.variable_scope('architecture'):
                self.h_lstm = RNN(self.I, self.sequence_length, self.num_hidden_lstm)

                # graph for training and predicting from strings
                self.logit = self.graph_embedding_to_reward_logit(embedding=self.h_lstm, reuse=False)

                # graph for training and predicting from precomputed embeddings
                self.logit_from_precomputed_embedding = self.graph_embedding_to_reward_logit(
                    embedding=self.precomputed_h_lstm, reuse=True)

                with tf.variable_scope('prediction'):
                    self.pred_proba = self.logit
                    self.pred_proba_from_precomputed_embedding = self.logit_from_precomputed_embedding
                self.pred = tf.round(self.pred_proba)
                self.pred_from_precomputed_embedding = tf.round(self.pred_proba_from_precomputed_embedding)

                with tf.variable_scope('optimization'):
                    self.cost = tf.reduce_mean(
                        -tf.log(self.pred_proba) * self.Y + (1 - self.Y) * (-tf.log(1 - self.pred_proba)))
                    tf.summary.scalar('loss', tensor=self.cost)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

                with tf.variable_scope('metrics'):
                    correct_pred = tf.equal(self.pred, self.Y)
                self.acc = tf.reduce_sum(tf.cast(correct_pred, "float")) / tf.cast(
                    tf.size(correct_pred), 'float')

                tp, fp, tn, fn = compute_confusion(self.pred, self.Y)

                self.prec = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fp, 'float')))
                self.recall = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fn, 'float')))

                tf.summary.scalar('accuracy', self.acc)

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

        # def

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return res

    def graph_embedding_to_reward_logit(self, embedding, reuse=False):
        # now use he init
        h_concat = tf.concat([self.S, embedding], axis=1)
        logits = tf.nn.sigmoid(nn(h_concat, [self.ff_size, 1], name="reward_scene", reuse=reuse))
        return logits

    def get_pred_from_precomputed_embedding(self):
        return self.pred_from_precomputed_embedding


class RewardFunctionAttention(TFRewardFunction):
    '''
    The flat attention (FA) model of the reward function
    '''
    def __init__(self, state_size, voc_size, sequence_length, batch_size, learning_rate, ff_size, num_hidden_lstm):
        super().__init__(state_size, voc_size, sequence_length)
        self.learning_rate = learning_rate
        self.beta = 0.00
        self.batch_size = batch_size
        self.ff_size = ff_size
        self.num_hidden_lstm = num_hidden_lstm  # self.sub_state_size

        # creating the graph
        with tf.variable_scope('reward'):
            with tf.variable_scope('inputs'):
                self.S = tf.placeholder("float", [None, self.state_size], name='state')
                self.I = tf.placeholder("float", [None, self.sequence_length, self.voc_size], name='instruction')
                self.Y = tf.placeholder("float", [None, 1], name='reward')
                self.precomputed_h_lstm = tf.placeholder("float", [None, self.num_hidden_lstm],
                                                         name='precomputed_hlstm')

            with tf.variable_scope('architecture'):
                self.h_lstm = RNN(self.I, self.sequence_length, self.num_hidden_lstm)

                # graph for training and predicting from strings
                self.logit, self.h_cast = self.graph_embedding_to_reward_logit(embedding=self.h_lstm, reuse=False)

                # graph for training and predicting from precomputed embeddings
                self.logit_from_precomputed_embedding, _ = self.graph_embedding_to_reward_logit(
                    embedding=self.precomputed_h_lstm, reuse=True)

                with tf.variable_scope('prediction'):
                    self.pred_proba = self.logit
                    self.pred_proba_from_precomputed_embedding = self.logit_from_precomputed_embedding
                self.pred = tf.round(self.pred_proba)
                self.pred_from_precomputed_embedding = tf.round(self.pred_proba_from_precomputed_embedding)

                with tf.variable_scope('optimization'):
                    self.cost = tf.reduce_mean(
                        -tf.log(self.pred_proba) * self.Y + (1 - self.Y) * (-tf.log(1 - self.pred_proba)))
                    tf.summary.scalar('loss', tensor=self.cost)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

                with tf.variable_scope('metrics'):
                    correct_pred = tf.equal(self.pred, self.Y)
                self.acc = tf.reduce_sum(tf.cast(correct_pred, "float")) / tf.cast(
                    tf.size(correct_pred), 'float')

                tp, fp, tn, fn = compute_confusion(self.pred, self.Y)

                self.prec = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fp, 'float')))
                self.recall = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fn, 'float')))

                tf.summary.scalar('accuracy', self.acc)

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

        # def

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return res

    def graph_embedding_to_reward_logit(self, embedding, reuse=False):
        # now use he init
        h_cast = tf.nn.sigmoid(nn(embedding, [self.state_size], name="embedding_cast", reuse=reuse))
        h_attention = tf.multiply(self.S, h_cast)
        logits = tf.nn.sigmoid(nn(h_attention, [self.ff_size, 1], name="reward_scene", reuse=reuse))
        return logits, h_cast

    def get_pred_from_precomputed_embedding(self):
        return self.pred_from_precomputed_embedding


class RewardFunctionCastAttentionShareOr(TFRewardFunction):
    '''
    The modular attention (MA) reward function model
    '''
    def __init__(self, or_params_path, body_size, obj_size, n_obj, state_size, voc_size, sequence_length,
                 batch_size, learning_rate, ff_size, num_hidden_lstm):
        super().__init__(state_size, voc_size, sequence_length)
        self.state_size = state_size
        self.voc_size = voc_size
        self.sequence_length = sequence_length

        self.learning_rate = learning_rate
        self.beta = 0.00
        self.batch_size = batch_size

        self.ff_size = ff_size

        self.body_indices = [i for i in range(body_size)]
        self.delta_body_indices = [i + self.state_size // 2 for i in range(body_size)]
        self.objs_indices = []
        self.delta_objs_indices = []
        for i in range(n_obj):
            self.objs_indices.append([j for j in range(body_size + i * obj_size, body_size + i * obj_size + obj_size)])
            self.delta_objs_indices.append([j + self.state_size // 2 for j in
                                            range(body_size + i * obj_size, body_size + i * obj_size + obj_size)])
            self.half_o = self.state_size // 2

        self.sub_state_size = 2 * (body_size + obj_size)
        self.num_hidden_lstm = num_hidden_lstm
        self.n_obj = n_obj
        self.or_params_path = or_params_path

        # creating the graph
        with tf.variable_scope('reward'):
            with tf.variable_scope('inputs'):
                self.S = tf.placeholder("float", [None, self.state_size], name='state')
                self.I = tf.placeholder("float", [None, self.sequence_length, self.voc_size], name='instruction')
                self.Y = tf.placeholder("float", [None, 1], name='reward')
                self.input_or = tf.placeholder("float", [None, self.n_obj], name="or_inputs")
                self.precomputed_h_lstm = tf.placeholder("float", [None, self.num_hidden_lstm], name='precomputed_hlstm')

            with tf.variable_scope('or_module'):
                self.or_operator = OrFunction(self.or_params_path[self.n_obj])
                self.out_or = self.or_operator.forward(self.input_or)
                # assert self.or_operator.n_obj == self.n_obj

            with tf.variable_scope('architecture'):
                self.h_lstm = RNN(self.I, self.sequence_length, self.num_hidden_lstm)

                # graph for training and predicting from strings
                self.logit, self.h_cast = self.graph_embedding_to_reward_logit(embedding=self.h_lstm, reuse=False)

                # graph for training and predicting from precomputed embeddings
                self.logit_from_precomputed_embedding, _ = self.graph_embedding_to_reward_logit(embedding=self.precomputed_h_lstm, reuse=True)

                with tf.variable_scope('prediction'):
                    self.pred_proba = self.logit
                    self.pred_proba_from_precomputed_embedding = self.logit_from_precomputed_embedding
                self.pred = tf.round(self.pred_proba)
                self.pred_from_precomputed_embedding = tf.round(self.pred_proba_from_precomputed_embedding)

                with tf.variable_scope('optimization'):
                    self.cost = tf.reduce_mean(
                        -tf.log(self.pred_proba) * self.Y + (1 - self.Y) * (-tf.log(1 - self.pred_proba)))
                    tf.summary.scalar('loss', tensor=self.cost)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

                with tf.variable_scope('metrics'):
                    correct_pred = tf.equal(self.pred, self.Y)
                self.acc = tf.reduce_sum(tf.cast(correct_pred, "float")) / tf.cast(
                    tf.size(correct_pred), 'float')

                tp, fp, tn, fn = compute_confusion(self.pred, self.Y)

                self.prec = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fp, 'float')))
                self.recall = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fn, 'float')))

                tf.summary.scalar('accuracy', self.acc)


    def set_or_function(self, n_obj):
        self.or_operator = OrFunction(self.or_params_path[n_obj])
        self.n_obj = n_obj

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

    # def

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return res


    def graph_embedding_to_reward_logit(self, embedding, reuse=False):
        # now use he init
        h_cast = tf.nn.sigmoid(nn(embedding, [self.sub_state_size], name="embedding_cast", reuse=reuse))

        # extract inputs for each object (and their delta)
        object_features = []
        for i in range(self.n_obj):
            obj_features = tf.concat(axis=1, values=[self.S[:, self.objs_indices[i][0]: self.objs_indices[i][-1] + 1],
                                                     self.S[:, self.objs_indices[i][0] + self.half_o: self.objs_indices[i][-1] + 1 + self.half_o]])
            object_features.append(obj_features)
        # extract body inputs and its delta
        state_body = tf.concat(axis=1, values=[self.S[:, :self.objs_indices[0][0]],
                                               self.S[:, self.half_o: self.half_o + self.objs_indices[0][0]]])

        # compute hadamard products with their respective attention (same attention for all objects)
        object_features_attention = []
        for i in range(self.n_obj):
            obj_features_att = tf.multiply(object_features[i], h_cast[:, int(state_body.shape[1]):])
            object_features_attention.append(obj_features_att)
        state_body_attention = tf.multiply(state_body, h_cast[:, :int(state_body.shape[1])])

        # compute the reward output for each object
        logits_obj = []
        for i in range(self.n_obj):
            input_reward_func = tf.concat(axis=1, values=[state_body_attention, object_features_attention[i]])
            reward_current = tf.nn.sigmoid(nn(input_reward_func, [self.ff_size, 1], name="reward_per_obj", reuse=reuse or i > 0))
            logits_obj.append(reward_current)
        logits_concat = tf.concat(axis=1, values=logits_obj)
        logits = self.or_operator.forward(logits_concat)

        return logits, h_cast


    def get_attention_vector(self):
        return self.h_cast

    def get_pred_from_precomputed_embedding(self):
        return self.pred_from_precomputed_embedding



class RewardFunctionCastAttentionShareMax(TFRewardFunction):
    def __init__(self, or_params_path, body_size, obj_size, n_obj, state_size, voc_size, sequence_length,
                 batch_size, learning_rate, ff_size, num_hidden_lstm):
        super().__init__(state_size, voc_size, sequence_length)
        self.state_size = state_size
        self.voc_size = voc_size
        self.sequence_length = sequence_length

        self.learning_rate = learning_rate
        self.beta = 0.00
        self.batch_size = batch_size

        self.ff_size = ff_size

        self.body_indices = [i for i in range(body_size)]
        self.delta_body_indices = [i + self.state_size // 2 for i in range(body_size)]
        self.objs_indices = []
        self.delta_objs_indices = []
        for i in range(n_obj):
            self.objs_indices.append([j for j in range(body_size + i * obj_size, body_size + i * obj_size + obj_size)])
            self.delta_objs_indices.append([j + self.state_size // 2 for j in
                                            range(body_size + i * obj_size, body_size + i * obj_size + obj_size)])
            self.half_o = self.state_size // 2

        self.sub_state_size = 2 * (body_size + obj_size)
        self.num_hidden_lstm = num_hidden_lstm
        self.n_obj = n_obj
        self.or_params_path = or_params_path

        # creating the graph
        with tf.variable_scope('reward'):
            with tf.variable_scope('inputs'):
                self.S = tf.placeholder("float", [None, self.state_size], name='state')
                self.I = tf.placeholder("float", [None, self.sequence_length, self.voc_size], name='instruction')
                self.Y = tf.placeholder("float", [None, 1], name='reward')
                self.input_or = tf.placeholder("float", [None, self.n_obj], name="or_inputs")
                self.precomputed_h_lstm = tf.placeholder("float", [None, self.num_hidden_lstm], name='precomputed_hlstm')


            with tf.variable_scope('architecture'):
                self.h_lstm = RNN(self.I, self.sequence_length, self.num_hidden_lstm)

                # graph for training and predicting from strings
                self.logit, self.h_cast = self.graph_embedding_to_reward_logit(embedding=self.h_lstm, reuse=False)

                # graph for training and predicting from precomputed embeddings
                self.logit_from_precomputed_embedding, _ = self.graph_embedding_to_reward_logit(embedding=self.precomputed_h_lstm, reuse=True)

                with tf.variable_scope('prediction'):
                    self.pred_proba = self.logit
                    self.pred_proba_from_precomputed_embedding = self.logit_from_precomputed_embedding
                self.pred = tf.round(self.pred_proba)
                self.pred_from_precomputed_embedding = tf.round(self.pred_proba_from_precomputed_embedding)

                with tf.variable_scope('optimization'):
                    self.cost = tf.reduce_mean(
                        -tf.log(self.pred_proba) * self.Y + (1 - self.Y) * (-tf.log(1 - self.pred_proba)))
                    tf.summary.scalar('loss', tensor=self.cost)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

                with tf.variable_scope('metrics'):
                    correct_pred = tf.equal(self.pred, self.Y)
                self.acc = tf.reduce_sum(tf.cast(correct_pred, "float")) / tf.cast(
                    tf.size(correct_pred), 'float')

                tp, fp, tn, fn = compute_confusion(self.pred, self.Y)

                self.prec = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fp, 'float')))
                self.recall = tf.reduce_sum(tf.cast(tp, 'float')) / (
                        tf.reduce_sum(tf.cast(tp, 'float')) + tf.reduce_sum(tf.cast(fn, 'float')))

                tf.summary.scalar('accuracy', self.acc)



    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

    # def

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return res


    def graph_embedding_to_reward_logit(self, embedding, reuse=False):
        # now use he init
        h_cast = tf.nn.sigmoid(nn(embedding, [self.sub_state_size], name="embedding_cast", reuse=reuse))

        # extract inputs for each object (and their delta)
        object_features = []
        for i in range(self.n_obj):
            obj_features = tf.concat(axis=1, values=[self.S[:, self.objs_indices[i][0]: self.objs_indices[i][-1] + 1],
                                                     self.S[:, self.objs_indices[i][0] + self.half_o: self.objs_indices[i][-1] + 1 + self.half_o]])
            object_features.append(obj_features)
        # extract body inputs and its delta
        state_body = tf.concat(axis=1, values=[self.S[:, :self.objs_indices[0][0]],
                                               self.S[:, self.half_o: self.half_o + self.objs_indices[0][0]]])

        # compute hadamard products with their respective attention (same attention for all objects)
        object_features_attention = []
        for i in range(self.n_obj):
            obj_features_att = tf.multiply(object_features[i], h_cast[:, int(state_body.shape[1]):])
            object_features_attention.append(obj_features_att)
        state_body_attention = tf.multiply(state_body, h_cast[:, :int(state_body.shape[1])])

        # compute the reward output for each object
        logits_obj = []
        for i in range(self.n_obj):
            input_reward_func = tf.concat(axis=1, values=[state_body_attention, object_features_attention[i]])
            reward_current = tf.nn.sigmoid(nn(input_reward_func, [self.ff_size, 1], name="reward_per_obj", reuse=reuse or i > 0))
            logits_obj.append(reward_current)
        logits_concat = tf.concat(axis=1, values=logits_obj)
        logits = tf.reduce_max(logits_concat, axis=1)
        return logits, h_cast


    def get_attention_vector(self):
        return self.h_cast

    def get_pred_from_precomputed_embedding(self):
        return self.pred_from_precomputed_embedding

