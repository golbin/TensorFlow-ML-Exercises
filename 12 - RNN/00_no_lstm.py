import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

sample = " if you want you"

char_set = list(set(sample))  # id -> char
char_dic = {w: i for i, w in enumerate(char_set)}

# settings
rnn_hidden_size = dic_size = len(char_dic)  # output size of each cell
batch_size = 1  # one sample data,one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)

sample_idx = [char_dic[c] for c in sample]  # char to index
x_data = tf.one_hot(sample_idx[:-1], dic_size)  # one hot
y_data = sample_idx[1:]

# split to input (char)length. This will decide unrolling size
x_split = tf.split(value=x_data, num_or_size_splits=[sequence_length])
outputs = x_split # No lstm

# softmax layer
softmax_w = tf.get_variable("softmax_w", [sequence_length, dic_size])
softmax_b = tf.get_variable("softmax_b", [dic_size])
outputs = outputs * softmax_w + softmax_b

outputs = tf.reshape(outputs, [batch_size, sequence_length, dic_size])
y_data = tf.reshape(y_data, [batch_size, sequence_length])
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(outputs, y_data, weights)
mean_loss = tf.reduce_mean(sequence_loss) / batch_size
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train_op, mean_loss])
    results = sess.run(outputs)
    for result in results:
        index = np.argmax(result, axis=1)
        print(''.join([char_set[t] for t in index]), l)
