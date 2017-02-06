import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tensorflow as tf
import mnist_cnn

from PIL import Image

imageDir = '/home/mhkim/data/images'

summary = '/home/mhkim/data/summaries/image2'

if tf.gfile.Exists(summary):
    tf.gfile.DeleteRecursively(summary)
tf.gfile.MakeDirs(summary)

#if os.path.exists(summary) == False : os.mkdir(summary)

img1 = Image.open(os.path.join(imageDir, 'number_font.png'))

SEED = 66478  # Set to None for random seed.
NUM_LABELS = 10

_width = img1.size[0]
_height = img1.size[1]
_basis = np.min(img1.size)
_widthPadding = 0
_heightPadding = 0

if _width % _basis != 0 :
    _widthPadding = _width // _basis
if _height % _basis != 0 :
    _heightPadding = _height // _basis

_im = Image.new("RGB", (_width + _widthPadding, _height + _heightPadding), "white")

_im.paste(img1, (0,0))

_pix = _im.load()

_width = _im.size[0]
_height = _im.size[1]

_shiftWidth = int(_width / _basis)
_shiftHeight = int(_height / _basis)

_batchSize = _shiftWidth * _shiftHeight

images = []
for row in range(_shiftHeight) :
    for cell in range(_shiftWidth):
        cropImage = _im.crop((cell * _basis, row * _basis , (cell+1)*_basis , (row + 1) * _basis))
        pixel = cropImage.load()
        cropImage = []
        for x in range(_basis) :
            cropImage.append([ [round(0.2126 * pixel[y, x][0] + 0.7152 * pixel[y, x][1] + 0.0722 * pixel[y, x][2])] for y in range(_basis)])
        images.append(cropImage)

X = tf.placeholder(tf.float32 , name='image_node')

W1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1, seed=SEED, dtype=tf.float32), name='weight_1')
B1 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='bias_1')

W2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32), name='weight_2')
B2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias_2')

fc1_weight = tf.Variable(tf.truncated_normal(shape=[_basis // 4 * _basis // 4 * 64, 512], stddev=0.1 , seed=SEED, dtype=tf.float32), name='fc1_weight')
fc1_bias = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_bias')

fc2_weight = tf.Variable(tf.truncated_normal(shape=[NUM_LABELS], stddev=0.1 , seed=SEED, dtype=tf.float32), name='fc2_weight')
fc2_bias = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32) , name='fc2_bias')

with tf.name_scope('model') :
    conv1 = tf.nn.conv2d(X , W1, strides=[1,1,1,1], padding='SAME', name='conv1')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, B1), name='relu1')
    pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME', name='pool2')

#    conv2 = tf.nn.conv2d(X, W2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
#    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, B2), name='relu1')
#    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    pool_shape = pool1.get_shape().as_list()

#    reshape = tf.reshape(pool2, [pool_shape[0] , pool_shape[1] * pool_shape[2] * pool_shape[3]])

 #   hidden = tf.nn.relu(tf.matmul(reshape, fc1_weight) + fc1_bias)

  #  hidden = tf.nn.dropout(hidden, 1. , seed=SEED)

   # logits = tf.matmul(hidden, fc2_weight) + fc2_bias



eval = tf.nn.softmax(pool1, name='eval')

tf.summary.scalar('eval_2', eval)

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(summary, sess.graph)

tf.global_variables_initializer().run()

tfImg = tf.summary.image('image1', X)

summaryEval , summary = sess.run([ eval , tfImg ], feed_dict={X:images})


writer.add_summary(summary)

writer.add_summary(summaryEval)



writer.close()

sess.close()


