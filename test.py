import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/", one_hot=True)

batch_xs , batch_ys = mnist.train.next_batch(10)
test_xs , test_ys = mnist.test.next_batch(10)

#print(len(batch_xs[0]))

def printMNIST ( ditis , lables ) :
    for n in range(len(ditis)) :
        _index = 0
        print([ i for i in range(len(lables[n])) if lables[n][i] > 0. ])

        for i in range(28):
            _v2 = []
            for j in range(28) :
                if ditis[n][_index] > 0. :
                    #value = '%f' % ( int(batch_xs[0][_index] * 100) / 100 )
                    value = '1'
                else :
                    value = ' '
                _v2.append(value)
                _index = _index + 1
            print(_v2)


printMNIST ( batch_xs , batch_ys )

print ( '{:_<150}\n' .format('_') )

printMNIST ( test_xs , test_ys )

#plt.plot(_v1 , 'o', label = 'MLP Training phase')
#plt.legend()
#plt.show()

#print(batch_ys)
'''

'''
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01) .minimize(cross_entropy)


init = tf.global_variables_initializer()


array = []
with tf.Session() as sess :

    sess.run(init)

# learning
    for i in range(1000) :
        batch_xs , batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict= { x : batch_xs, y_:batch_ys})

#validation
    correct_prediction = tf.equal( tf.argmax(y, 1) , tf.argmax(y_, 1) )

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = { x:mnist.test.images, y_:mnist.test.labels})
    print ( 'result : ' , result )

#print ( len(array) )
#plt.plot(array)
#plt.show()

