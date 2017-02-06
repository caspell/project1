import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor.txt', unpack=True)

x_data = xy[0:-1]
y_data = xy[-1]

print (x_data)
print (y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1. , 1.))

h = tf.matmul(W, X)

hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

learning_rate = tf.Variable(0.01)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess :
    tf.global_variables_initializer().run()
    for step in range(1000) :
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200  == 0 :
            print ( step , sess.run(cost, feed_dict={ X:x_data, Y:y_data }) , sess.run(W) )

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ( sess.run([hypothesis, tf.floor(hypothesis + 0.5 ), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print ("Accuracy : " , accuracy.eval({X:x_data, Y:y_data }))
