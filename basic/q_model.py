import collections
from path import tf_data_dir

import tensorflow as tf
slim = tf.contrib.slim

import tf_utils

import random
import numpy as np

def hyperparams(**kwargs):
    params = collections.OrderedDict()
    params["task"] = "cartpole"
    params["model"] = "fc"
    params["layers"] = "10-10-5-1"
    params["batchSize"] = 64
    params["initlr"] = 1e-2
    params["trainIters"] = 10000
    params["trainEpisodes"] = 1000 # number of episodes between changing the network
    params["initckpt"] = None

    for arg in kwargs:
        assert arg in params
        params[arg] = kwargs[arg]

    return params

def dict_to_string(params):
    name = ""
    for key in params:
        if params[key] is not None:
            name = name + str(key) + "_" + str(params[key]) + "_"
    return name[:-1]

def init_weights(name, shape):
    print "initializing", name, shape
    return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(0, 10.0))

def fc_string(network_string):
    return map(int, network_string.split("-"))

def make_network(x, network_size):
    """Makes fully connected network with input x and given layer sizes.
    Assume len(network_size) >= 2
    """
    input_size = network_size[0]
    output_size = network_size.pop()
    a = input_size
    cur = x
    i = 0
    for a, b in zip(network_size, network_size[1:]):
        W = init_weights("W" + str(i), [a, b])
        B = init_weights("B" + str(i), [1, b])
        cur = tf.sigmoid(tf.matmul(cur, W) + B)
        i += 1
    W = init_weights("W" + str(i), [b, output_size])
    B = init_weights("B" + str(i), [1, output_size])
    prediction = tf.matmul(cur, W) + B
    return prediction

input_size_lookup = {'cartpole': 4, }
action_size_lookup = {'cartpole': 2, }

class QModel(object):
    def __init__(self, train_params, test_params=None, erase_model=False):
        self.name = dict_to_string(train_params)
        self.network = tf_utils.TFNet(self.name,
            logDir= tf_data_dir + 'tf_logs/',
            modelDir= tf_data_dir + 'tf_models/',
            outputDir= tf_data_dir + 'tf_outputs/',
            eraseModels=erase_model)

        self.params = train_params
        # some logic missing here to load the right model and right test-time params

        self.input_dim = input_size_lookup[self.params["task"]]
        self.action_dim = action_size_lookup[self.params["task"]]

        self.X = tf.placeholder("float", [None, self.input_dim])
        self.U = tf.placeholder("float", [None, self.action_dim])
        self.Y = tf.placeholder("float", [None, 1])

        x = tf.concat(1, [self.X, self.U])

        with slim.arg_scope([slim.fully_connected],
            weights_initializer=tf.contrib.layers.xavier_initializer,
            weights_regularizer=slim.l2_regularizer(0.0005),
            activation_fn=tf.nn.elu):
            # self.layers = [self.params["hdim"]] * self.params["layers"] + [1]
            # self.y = slim.stack(x, slim.fully_connected, self.layers, scope="fc")
            d = self.input_dim + self.action_dim
            self.y = make_network(x, [d] + fc_string(self.params["layers"]))

        self.loss = tf.nn.l2_loss(self.Y - self.y)

        self.network.add_to_losses(self.loss)

        self.sess = tf.Session()
        # self.network.restore_model(self.sess)
        tf.initialize_all_variables().run(session=self.sess)

        self.inputs = [self.X, self.U, self.Y]
        self.train_network = tf_utils.TFTrain(self.inputs, self.network, batchSz=self.params["batchSize"], initLr=self.params["initlr"] or 0.001)
        self.train_network.add_loss_summaries([self.loss], ['loss'])

    def train(self, data, max_iters = 10000):
        """Data is a list of (s, a, t)
        s: state
        a: action
        t: target, r + gamma * max Q(s', :)
        """
        random.shuffle(data)
        batch_size = self.params["batchSize"]

        N = int(len(data) * 9 / 10)
        training_data = data[:N]
        test_data = data[N:]

        def get_batch(data):
            d = random.sample(data, batch_size)
            X = np.zeros((batch_size, self.input_dim))
            U = np.zeros((batch_size, self.action_dim))
            Y = np.zeros((batch_size, 1))
            for i in range(batch_size):
                s, a, t = d[i]
                X[i, :] = s
                U[i, a] = 1
                Y[i, :] = t

            return X, U, Y

        def train_batch(inputs, batch_size, isTrain):
            if isTrain:
                X, U, Y = get_batch(training_data)
            else:
                X, U, Y = get_batch(test_data)
            feed_dict = {self.X: X, self.U: U, self.Y: Y}
            
            return feed_dict

        self.train_network.maxIter_ = self.params["trainIters"]
        self.train_network.dispIter_ = 1000
        self.train_network.saveIter_ = 11000
        self.train_network.train(train_batch, train_batch) #, gpu_fraction=0.95, use_existing=use_existing, init_path=self.init_path)

    def f(self, s):
        """Returns the Q(s, *)"""
        b = self.action_dim
        x = np.zeros((b, self.input_dim))
        u = np.zeros((b, self.action_dim))
        y = np.zeros((b, 1))
        for i in range(b):
            x[i, :] = s
            u[i, i] = 1
        feed_dict = {self.X: x, self.U: u, self.Y: y}
        result = self.sess.run([self.y], feed_dict)
        y = result[0]
        # print "x", x
        # print "u", u
        # print "y", y[:, 0]
        return y[:, 0]