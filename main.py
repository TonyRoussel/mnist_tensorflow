#!/usr/bin/env python2.7

import input_data
import tensorflow as tf


# "The role of the Python code is therefore to build this external computation graph,
# and to dictate which parts of the computation graph should be run"


# Data importation via google data import script.
# mnist is a class which store training, validation and testing sets as numpy arrays.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# A Session is use to  execute ops in the graph.
# Here we use an InteractiveSession to gain flexibility
# aka 'interleave operations which build a computation graph with ones that run the graph'
sess = tf.InteractiveSession()

# We create x and y_ placeholder, which will be later fed when we'll ask
# TensorFlow to run a computation.
# The input (x) will be a 2d tensor of floating point number. The shape parameter
# is not mandatory but we set it to catch bugs related to inconsistent tensor shapes.
# "784 is the dimensionality of a single flattened MNIST image, and None indicate that
# the first dimension, aka the batch size, can be of any size"
x = tf.placeholder("float", shape=[None, 784])
# "The output classes (y_) will also consist of a 2d tensor, where each row is a one-hot 10-dimensional vector"
y_ = tf.placeholder("float", shape=[None, 10])

# We define the weights W and biases b for our model. To handle it we use Variable,
# which "is a value that lives in TF computation graph. It can be used and even
# modified by the computation"
# Both parameter are initialize as tensor full of zeros.
# "W is a 784x10 matrix (because we have 784 input features and 10 outputs), and b is
# a 10-dimentional vector (because we have 10 classes).
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# To use Variable within a session, they must be initialized using that session.
# "This step takes the initial values (in this case tensors full of zeros) that have
# already been specified, and assigns them to each Variable. This can be done for all
# Variables at once."
sess.run(tf.initialize_all_variables())
