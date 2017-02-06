import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 4], -1.0,1.0 ), name='w1')
b1 = tf.Variable(tf.zeros([4]), name='Bias1')
W2 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w2')
b2 = tf.Variable(tf.zeros([4]), name='Bias2')
W3 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w3')
b3 = tf.Variable(tf.zeros([4]), name='Bias3')

W4 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w4')
b4 = tf.Variable(tf.zeros([4]), name='Bias4')
W5 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w5')
b5 = tf.Variable(tf.zeros([4]), name='Bias5')
W6 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w6')
b6 = tf.Variable(tf.zeros([4]), name='Bias6')
W7 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w7')
b7 = tf.Variable(tf.zeros([4]), name='Bias7')
W8 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w8')
b8 = tf.Variable(tf.zeros([4]), name='Bias8')
W9 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w9')
b9 = tf.Variable(tf.zeros([4]), name='Bias9')
W10 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name='w10')
b10 = tf.Variable(tf.zeros([4]), name='Bias10')

W11 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name='w11')
b11 = tf.Variable(tf.zeros([1]), name='Bias11')

'''
tf.summary.histogram('w1', W1)
tf.summary.histogram('b1', b1)
tf.summary.histogram('w2', W2)
tf.summary.histogram('b2', b2)
tf.summary.histogram('w3', W3)
tf.summary.histogram('b3', b3)
tf.summary.histogram('w4', W4)
tf.summary.histogram('b4', b4)
'''

with tf.name_scope('layer1') :
    #L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
with tf.name_scope('layer2') :
    #L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
with tf.name_scope('layer3') :
    #L3 = tf.sigmoid(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
with tf.name_scope('layer4') :
    #L4= tf.sigmoid(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
with tf.name_scope('layer5') :
    #L5= tf.sigmoid(tf.matmul(L4, W5) + b5)
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)

with tf.name_scope('layer6') :
    #L6= tf.sigmoid(tf.matmul(L5, W6) + b6)
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
with tf.name_scope('layer7') :
    #L7= tf.sigmoid(tf.matmul(L6, W7) + b7)
    L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
with tf.name_scope('layer8') :
    #L8= tf.sigmoid(tf.matmul(L7, W8) + b8)
    L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
with tf.name_scope('layer9') :
    #L9 = tf.sigmoid(tf.matmul(L8, W9) + b9)
    L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)
with tf.name_scope('layer10') :
    #L10 = tf.sigmoid(tf.matmul(L9, W10) + b10)
    L10 = tf.nn.relu(tf.matmul(L9, W10) + b10)
''''''
with tf.name_scope('layer11') :
    hypothesis  = tf.sigmoid(tf.matmul(L10, W11) + b11)

with tf.name_scope('cost') :
    cost = -tf.reduce_mean( Y * tf.log(hypothesis) + ( 1 - Y ) * tf.log(1-hypothesis) )
    tf.summary.scalar('cost', cost)

learning_rate = tf.Variable(0.05)
with tf.name_scope('train') :
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

with tf.Session() as sess :
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/mhkim/xor_logs3", sess.graph)
    for step in range(20000) :
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000  == 0 :
            print ( step , sess.run(cost, feed_dict={ X:x_data, Y:y_data }) )
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary , step)
            #tf.summary.scalar("accuracy", accuracy)

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    with tf.name_scope('result') :
        tf.summary.scalar('scala', accuracy)

    print ("Accuracy : " , accuracy.eval({X:x_data, Y:y_data }))
    #print ( sess.run([hypothesis, tf.floor(hypothesis + 0.5 ), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))

    #correct_prediction = tf.floor()
    #print ( hypothesis.eval(feed_dict={X:x_data}))


