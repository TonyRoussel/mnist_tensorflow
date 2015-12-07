#!/usr/bin/env python2.7

import input_data
import tensorflow as tf
import pprint as pp


# For this model we'll need "to create a lot of weights and biases"
# "One should generally initialize weights with a small amount of noise for 
# symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons,
# it is also good practice to initialize them with a slightly positive
# initial bias to avoid "dead neurons.""
# "Instead of doing this repeatedly while we build the model, let's create
# two handy functions to do it for us"
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Data importation via google data import script.
# mnist is a class which store training, validation and testing sets as numpy arrays.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()




sess.run(tf.initialize_all_variables())
for i in xrange(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print cross_entropy.eval({x: mnist.train.images, y_: mnist.train.labels}, sess)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict = {x: mnist.test.images, y_ : mnist.test.labels})
