from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/mhkim/data/mnist', one_hot=True)

import tensorflow as tf

# 1e-4
learning_rate = 0.001
#training_epochs 20000
training_epochs = 1000

# 50
batch_size = 50

display_step = 100

# 1024
n_hidden_1 = 256
n_hidden_2 = 256

n_input = 784
n_class= 10

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, n_input])
y_ = tf.placeholder(tf.float32, shape=[None, n_class])


W = tf.Variable(tf.zeros([n_input,n_class]))
b = tf.Variable(tf.zeros([n_class]))


sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b)


W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))


x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Second layer
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Densely Connected Layer
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, n_hidden_1], stddev=0.1))

b_fc1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_1]))


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout layer
W_fc2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_class], stddev=0.1))

b_fc2 = tf.Variable(tf.constant(0.1, shape=[n_class]))

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Train and Evaluate the Model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(training_epochs):
    batch = mnist.train.next_batch(batch_size)
    if i % display_step == 0:
        train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))