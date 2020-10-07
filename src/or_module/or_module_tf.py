import tensorflow as tf
import pickle

path_to_dict = ''
class OrFunction:

    # Create model
    def __init__(self, path_to_dict=path_to_dict):
        with open(path_to_dict, 'rb') as f:
            self.data = pickle.load(f)
        self.layers_sizes = self.data['layers']
        self.nb_layers = len(self.layers_sizes)
        self.n_obj = self.layers_sizes[0][0]
        self.layers = []
        for i in range(self.nb_layers):
            self.layers.append(self.create_layer(layer_id=i,
                                                 size=self.layers_sizes[i],
                                                 weight=self.data['fc{}_weight'.format(i+1)],
                                                 bias=self.data['fc{}_bias'.format(i+1)]))
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
        return x

    def create_layer(self, layer_id, size, weight, bias):
        weight_init = tf.constant(weight.transpose())
        bias_init = tf.constant(bias)
        W = tf.get_variable('or_l{}_w'.format(layer_id), dtype=tf.float32, initializer=weight_init, trainable=False)
        b = tf.get_variable('or_l{}_b'.format(layer_id), dtype=tf.float32, initializer=bias_init, trainable=False)
        def fc(x):
            return tf.nn.bias_add(tf.matmul(x, W), b)
        return fc



import numpy as np
n_points = 100000
n_objs = 3
# balanced test set
x_test = np.random.uniform(0, 1, [n_points, n_objs])
y_test = (x_test > 0.5).astype(np.int).max(axis=1)
x_test_pos = x_test[y_test == 1][:n_points // 2]
y_test_pos = y_test[y_test == 1][:n_points // 2]
x_test_neg = np.random.uniform(0, 0.5, [n_points // 2, n_objs])
y_test_neg = np.array([0] * (n_points // 2))
y_test = np.concatenate([y_test_pos, y_test_neg])
x_test = np.concatenate([x_test_pos, x_test_neg], axis=0)
assert (y_test == 0).sum() == n_points // 2

x = tf.placeholder(tf.float32, [None, n_objs])
sess = tf.InteractiveSession()
model = OrFunction(path_to_dict)
sess.run(tf.global_variables_initializer())
out = sess.run(model.forward(x), feed_dict={x:x_test})

test = (out.squeeze() > 0.5) == y_test
print(test.sum() / n_points)


stop = 1

