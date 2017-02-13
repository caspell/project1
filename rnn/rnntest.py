import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
char_dic = { w:i for i , w in enumerate(char_rdic)}

print(char_dic)

ground_truth = [char_dic[c] for c in 'hello']

print (ground_truth)

x_data = np.array([[1,0,0,0]
                  ,[0,1,0,0]
                  ,[0,0,1,0]
                  ,[0,0,1,0]], dtype='f')

x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)

print(x_data)

rnn_size = len(char_dic)
batch_size = 1
output_size= 4

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size, input_size=None)

print(rnn_cell)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)

print(initial_state)

initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size])

print(initial_state_1)

x_split = tf.split(0, len(char_dic), x_data)

print(x_split)

outputs , state = tf.nn.rnn(cell=rnn_cell,inputs=x_split,initial_state = initial_state)

print(outputs)
print(state)

logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])

logits.get_shape()

targets = tf.reshape(ground_truth[1:],[-1])

targets.get_shape()

weights= tf.ones([len(char_dic) * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[targets],[weights])

cost = tf.reduce_sum(loss) / batch_size

train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char_rdic[t] for t in result])

