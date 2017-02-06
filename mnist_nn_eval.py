import numpy as np
import tensorflow as tf
import os
import sys



from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/mhkim/data/mnist', one_hot=True)

train_checkpoint = '/home/mhkim/data/checkpoint/mnist_nn/save.ckpt'


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform :
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else :
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


x = tf.placeholder('float', [None, 784], name='X')
y = tf.placeholder('float', [None, 10], name='Y')

W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], initializer=xavier_init(256, 256))
W4 = tf.get_variable("W4", shape=[256, 10], initializer=xavier_init(256, 10))

b1 = tf.Variable(tf.random_normal([256]), name='bias1')
b2 = tf.Variable(tf.random_normal([256]), name='bias2')
b3 = tf.Variable(tf.random_normal([256]), name='bias3')
b4 = tf.Variable(tf.random_normal([10]), name='bias4')

dropout_rate = tf.placeholder("float", name='dropout_rate')

L1 = tf.nn.relu(tf.add(tf.matmul(x, W1,) , b1), name='relu1')
_L1 = tf.nn.dropout(L1, dropout_rate, name='relu1-dropout')
L2 = tf.nn.relu(tf.add(tf.matmul(_L1, W2,) , b2), name='relu2')
_L2 = tf.nn.dropout(L2, dropout_rate, name='relu2-dropout')
L3 = tf.nn.relu(tf.add(tf.matmul(_L2, W3,) , b3), name='relu3')
_L3 = tf.nn.dropout(L3, dropout_rate, name='relu3-dropout')
hypothesis = tf.add(tf.matmul(_L3, W4), b4, name='hypothesis')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y), name='cost')

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()

saver.restore(sess, train_checkpoint)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#prediction1 = hypothesis.eval({x: [mnist.test.images[0]] , dropout_rate: 1 })

#print ( prediction1 )

print ( mnist.test.images[0])
print ( tf.argmax(hypothesis, 1).eval({x: [mnist.test.images[0]] , dropout_rate: 1 }) )



print("Accuracy : ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1}))

print("Accuracy : ", accuracy.eval({x: [mnist.test.images[0]], y: [mnist.test.labels[0]], dropout_rate: 1}))

sess.close()
