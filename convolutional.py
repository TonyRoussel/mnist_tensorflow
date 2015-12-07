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


# "TensorFlow also gives us a lot of flexibility in convolution and pooling operations.
# How do we handle the boundaries? What is our stride size?
# In this example, we're always going to choose the vanilla version.
# Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
# Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner,
# let's also abstract those operations into functions."
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# "We can now implement our first layer
# It will consist of convolution, followed by max pooling. The convolutional will compute
# 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32].
# The first two dimensions are the patch size, the next is the number of input channels,
# and the last is the number of output channels. We will also have a bias vector with a
# component for each output channel."
# W_conv1 = weight_variable([patch_size_x, patch_size_y, num_input_channel, num_output_channel])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# "To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions 
# corresponding to image width and height, and the final dimension corresponding to the number of color channels."
x_image = tf.reshape(x, [-1, 28, 28, 1])

# "We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool."
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)




sess.run(tf.initialize_all_variables())
for i in xrange(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print cross_entropy.eval({x: mnist.train.images, y_: mnist.train.labels}, sess)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict = {x: mnist.test.images, y_ : mnist.test.labels})
