# This file creates the network architecture - is the ideal file to remake with EDEN for each new network

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
# import tflearn
import distutils.version
import pickle
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

# Othe input to make EDEN model work
# NN weights mean upon initialisation
weights_mean = 0.0
# NN weights standard deviation upon initialisation
weights_stdev = 0.01


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        print("in conv2d")
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def fully_conn(x, n_units, name, mean, stddev, dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        x = flatten(x)
        weights_shape = [int(x.shape[1]), n_units]
        # w = tf.get_variable("W", weights_shape, dtype, initializer=tf.truncated_normal(weights_shape, mean, stddev, dtype),
        #                     collections=collections)
        w = tf.get_variable("W", weights_shape, dtype, initializer=tf.glorot_normal_initializer(),
                            collections=collections)
        # b = tf.get_variable("b", [n_units], initializer=tf.constant_initializer(0.0),
        #                     collections=collections)
        b = tf.get_variable("b", shape=n_units, initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.xw_plus_b(x, w, b)

def max_pool_2d(x, kernel, name, stride = (1, 1), dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        # as in Keras, default to kernel_size
        # stride_shape = [1, stride[0], stride[1], 1]
        stride_shape = [1, kernel, kernel, 1]
        ksize_shape = [1, kernel, kernel, 1]

        return tf.nn.max_pool(x, ksize_shape, stride_shape, padding = 'SAME')

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keepdims=True), 1), [1])
    return tf.one_hot(value, d)

def activation_function(chromosome_layer_activation_name, net):
    if chromosome_layer_activation_name == 'linear':
        return net
    else:
        activation_func = getattr(tf.nn, chromosome_layer_activation_name)
        return activation_func(net)

#add names?
def build_model(net, chromosome):
    try:

        """ Build the NN model.

        Args:
            chromosome: the chromosome

        Returns:
            None
        """
        net = net

        # tf.reset_default_graph()
        # with tf.Graph().as_default(), tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     tflearn.config.init_training_mode()

            # Input to neural net
            # -------------------
            # Here we call the `construction_input_shape` function which will generate the correct
            # input for the network. This is based on X_train.
        # net = tflearn.input_data(shape=construct_input_shape())

        # First find out how many layers this configuration has
        number_of_layers = len(chromosome[6])

        # Iterate over each gene in the chromosome which corresponds
        # to the layers
        for i in range(0, number_of_layers):

            # Get the type of configuration
            layer_name = chromosome[6][i].get('name')

            # -----------------
            # conv_2d
            # -----------------
            # if layer_name == 'conv_2d':
            #     net = tflearn.layers.conv_2d(net, nb_filter=chromosome[6][i].get('nb_filter'),
            #                                  filter_size=chromosome[6][i].get('filter_size'),
            #                                  activation=chromosome[6][i].get('activation'),
            #                                  regularizer="L2",
            #                                  name = layer_name + "-l{}".format(i + 1))

            # Build conv2d in same way os expected A3C code
            # Attempt directly below, and original function (comment out in run!) below that
            if layer_name == 'conv_2d':
                # activation_function = getattr(tf.nn, chromosome[6][i].get('activation'))
                net = activation_function(chromosome[6][i].get('activation'),
                                            conv2d(net,
                                                    num_filters=chromosome[6][i].get('nb_filter'),
                                                    name="l{}".format(i + 1),
                                                    filter_size=[chromosome[6][i].get('filter_size'), chromosome[6][i].get('filter_size')],
                                                    stride=[2, 2])) #using default stride (1, 1) in tflearn


            # -----------------
            # conv_1d
            # -----------------
            # elif layer_name == 'conv_1d':
            #     net = tflearn.layers.conv_1d(net, nb_filter=chromosome[6][i].get('nb_filter'),
            #                                  filter_size=chromosome[6][i].get('filter_size'),
            #                                  activation=chromosome[6][i].get('activation'),
            #                                  regularizer="L2",
            #                                  name = layer_name + "-l{}".format(i + 1))

            # -----------------
            # max_pool_1d
            # -----------------
            # elif layer_name == 'max_pool_1d':
            #     net = tflearn.layers.max_pool_1d(net,
            #                                      kernel_size=chromosome[6][i].get('kernel_size'),
            #                                      name = layer_name + "-l{}".format(i + 1))

            # -----------------
            # max_pool_2d
            # -----------------
            elif layer_name == 'max_pool_2d':
                net = max_pool_2d(net,
                                    kernel=chromosome[6][i].get('kernel_size'),
                                    name="l{}".format(i + 1))

            # -----------------
            # fully_conn
            # -----------------
            elif layer_name == 'fully_conn':
                # net = tflearn.fully_connected(net, n_units=chromosome[6][i].get('units'),
                #                               activation=chromosome[6][i].get('activation'),
                #                               weights_init=tflearn.initializations.truncated_normal(
                #                                   mean=weights_mean, stddev=weights_stdev,
                #                                   dtype=tf.float32),
                #                               name = layer_name + "-l{}".format(i + 1))
                # activation_function = getattr(tf.nn, chromosome[6][i].get('activation'))
                net = activation_function(chromosome[6][i].get('activation'),
                                            fully_conn(net,
                                                        n_units=chromosome[6][i].get('units'),
                                                        name="l{}".format(i + 1),
                                                        mean=weights_mean, stddev=weights_stdev))

            # -----------------
            # dropout
            # -----------------
            # elif layer_name == 'dropout':
            #     net = tflearn.layers.core.dropout(net,
            #                                       keep_prob=chromosome[6][i].get('keep_prob'),
            #                                       name = layer_name + "-l{}".format(i + 1))

            # -----------------
            # max_pool
            # -----------------
            else:
                print ('Error, else value reached in for i in range(0, number_of_layers):')
                return -1

        # Add hyper parameters

        # net = tflearn.regression(net, optimizer=chromosome[0],
        #                          learning_rate=chromosome[1],
        #                          loss='categorical_crossentropy')



    except Exception as e:
        print("Error caught (build_model): ", e)
        return -1

    return net

class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, mod_path):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        # add in generated model architecture here
        # for i in range(4):
        #     x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        #
        # # print(x)
        #
        # mod_path = "/Users/blakecuningham/Dropbox/MScDataScience/Thesis/EDEN/test_models/testmod.p"

        with open(mod_path, "rb") as filehandle:
            test_chromo_loaded = pickle.load(filehandle)

        x = build_model(x, test_chromo_loaded)
        # # print(x)

        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        size = 256
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        # First intiatilize the c and h cells, then create placeholders
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        # state input
        self.state_in = [c_in, h_in]

        # Establish state input cell
        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        # Build lstm with basic lstm cell, x, intial state of the lstm, and the step size of x
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)

        # Separate out the two lstm states
        lstm_c, lstm_h = lstm_state
        # add the lstm to the graph
        x = tf.reshape(lstm_outputs, [-1, size])

        # action space of a3c model
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))

        # value function of a3c model
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
