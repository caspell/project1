'''
added xavier_initializer
added dropout
'''
import numpy as np
import tensorflow as tf
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/mhkim/data/mnist', one_hot=True)

train_checkpoint = '/home/mhkim/data/checkpoint/mnist_nn/'

if tf.gfile.Exists(train_checkpoint):
    tf.gfile.DeleteRecursively(train_checkpoint)
tf.gfile.MakeDirs(train_checkpoint)

#learning_rate = 0.001
training_epochs = 15
display_step = 1
batch_size = 100

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

#activation = tf.nn.softmax(tf.matmul(x, W) + b)
#cost = tf.reduce_mean(-tf.reduce_sum(y* tf.log(activation) , reduction_indices=1))

dropout_rate = tf.placeholder("float", name='dropout_rate')

L1 = tf.nn.relu(tf.add(tf.matmul(x, W1,) , b1), name='relu1')
_L1 = tf.nn.dropout(L1, dropout_rate, name='relu1-dropout')
L2 = tf.nn.relu(tf.add(tf.matmul(_L1, W2,) , b2), name='relu2')
_L2 = tf.nn.dropout(L2, dropout_rate, name='relu2-dropout')
L3 = tf.nn.relu(tf.add(tf.matmul(_L2, W3,) , b3), name='relu3')
_L3 = tf.nn.dropout(L3, dropout_rate, name='relu3-dropout')
hypothesis = tf.add(tf.matmul(_L3, W4), b4, name='hypothesis')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y), name='cost')

#######

batch = tf.Variable(0, dtype=tf.float32, name='batch')

train_size = mnist.test.labels.shape[0]

global_step = tf.Variable(0, trainable=False, name='global_step')
starter_learning_rate = 0.001

learning_rate = tf.train.exponential_decay(
    starter_learning_rate,                # Base learning rate.
    global_step,            # Current index into the dataset.
    train_size,             # Decay step.
    0.96,                   # Decay rate.
    staircase=True,
    name='decay_learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost, name='train')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess :
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    for epoch in range(training_epochs) :
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs , batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={ x:batch_xs, y:batch_ys , dropout_rate:0.7})
            avg_cost += sess.run(cost, feed_dict={ x:batch_xs, y:batch_ys  , dropout_rate:0.7}) / total_batch
        if epoch % display_step == 0:
            print ( "Epoch : ", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost) , '%f' % learning_rate.eval() )

    saver.save(sess=sess, save_path=os.path.join(train_checkpoint, 'save.ckpt'))

    sys.stdout.flush()

    print ("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy : ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels , dropout_rate:1 }))