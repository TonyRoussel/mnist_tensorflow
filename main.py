#!/usr/bin/env python2.7

import input_data
import tensorflow as tf

# Data importation via google data import script.
# mnist is a class which store training, validation and testing sets as numpy arrays.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# A Session is use to  execute ops in the graph.
# Here we use an InteractiveSession to gain flexibility
# aka 'interleave operations which build a computation graph with ones that run the graph'
sess = tf.InteractiveSession()

