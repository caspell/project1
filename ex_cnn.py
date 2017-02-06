import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# MNIST 데이터 크기 ( 28 x 28 이미지 )
n_input = 784

# MNIST 데이터가 분류될 클래스
n_classes = 10

# 과적합 방지 ( 은닉 뉴런 , 입력력뉴런 , 출력 뉴런) 을 부분적으로 무시.
dropout = 0.75

## 출력 레이어
wout = tf.Variable(tf.random_normal([1024, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))

keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32 , [None , n_input])

# reshape 로 입력 데이터를 4 차원 행렬로 변환
# -1 , 너비 , 높이 , 컬러 채널의 개수 ( 여기선 흑백이므로 1, RGB 의 경우 3 )
_X = tf.reshape(x, shape=[-1, 28, 28 ,1])

y = tf.placeholder(tf.float32, [None, n_classes])

## 첫번째 합성곱 레이어
## 32 는 특징지도 개수
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))

## 첫번째 합성곱 레이어 생성
## padding = SAME : 출력 텐서값은 입력 텐서 값과 같은 크기를 갖는다는 의미
conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_X , wc1, strides=[1,1,1,1] ,padding='SAME'), bc1))

# 2x2 max polling 을 추행해 풀링 레이어에 있는 각 점들에서 정보를 요약
# 각각의 합성곱 레이어에 최댓값 풀링 함수를 적용하기 때문에 합성곱과 풀링을 수행하는 여러개의 레이어로 구성한다.
conv1 = tf.nn.max_pool(conv1 , ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# dropout 적용
conv1 = tf.nn.dropout(conv1, keep_prob)


wc2 = tf.Variable(tf.random_normal([5,5,32,64]))
bc2 = tf.Variable(tf.random_normal([64]))

conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1,1,1,1], padding='SAME') , bc2))
conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv2 = tf.nn.dropout(conv2, keep_prob)
'''
wc3 = tf.Variable(tf.random_normal([5,5,16,32]))
bc3 = tf.Variable(tf.random_normal([32]))

conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1,1,1,1], padding='SAME') , bc3))
conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv3 = tf.nn.dropout(conv3, keep_prob)

wc4 = tf.Variable(tf.random_normal([5,5,128,256]))
bc4 = tf.Variable(tf.random_normal([256]))

conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, wc4, strides=[1,1,1,1], padding='SAME') , bc4))
conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv4 = tf.nn.dropout(conv4, keep_prob)
'''

## 두번째 합성곱 레이어
# 합성곱 연산을 위해 공유 가중치와 공유 편향 값을 선언하고 초기화
'''
wc2 = tf.Variable(tf.random_normal([5,5,32,64]))
bc2 = tf.Variable(tf.random_normal([64]))

conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1,1,1,1], padding='SAME') , bc2))
conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv2 = tf.nn.dropout(conv2, keep_prob)
'''

## 완전 연결 레이어
wd1 = tf.Variable(tf.random_normal([ 7 * 7 * 64 , 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))

# 두번째 합성곱 레이어에서 받은 텐서를 변환해 벡터 형태의 배치로 만든다.
#dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
dense1 = tf.nn.dropout(dense1, keep_prob)

# 소프트맥스 함수를 적용하기 전에 이미지가 각 클래스에 속할 근거에 대해 계산해야 한다.
pred = tf.add(tf.matmul(dense1, wout), bout)

# 모델 학습 및 평가
# 근거 값은 10 개의 클래스 각각에 속할 확률로 변환한다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# 비용함수의 최적화 방법
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 텐서의 모델 평가
correct_pred = tf.equal(tf.argmax(pred,1 ), tf.argmax(y , 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


config = tf.ConfigProto(log_device_placement=True)

with tf.Session(config=config) as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)

    sess.run(init)
    step = 1
    while step * batch_size < training_iters :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = { x: batch_xs, y :batch_ys, keep_prob : dropout})
        if step % display_step == 0 :
            acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys, keep_prob : 1.})
            loss = sess.run(cost, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.})
            print ("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
        step +=1

    print ("Optimization Finished!")
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x:mnist.test.images[:256], y:mnist.test.labels[:256], keep_prob:1.}))






