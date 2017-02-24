import tensorflow as tf
from tensorflow.contrib import rnn

sample = " if you want you"
char_set = list(set(sample))  # id -> char
char_dic = {w: i for i, w in enumerate(char_set)}

# settings
rnn_hidden_size = dic_size = len(char_dic)  # output size of each cell
batch_size = 1  # one sample data,one batch
input_len = len(sample) - 1  # number of lstm rollings (unit #)

sample_idx = [char_dic[c] for c in sample]  # char to index
x_data = tf.one_hot(sample_idx[:-1], dic_size)  # one hot
y_data = sample_idx[1:]

# Make lstm with rnn_hidden_size (each unit input vector size)
lstm = rnn.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
lstm = rnn.MultiRNNCell([lstm] * 1, state_is_tuple=True)

# split to input (char)length. This will decide unrolling size
x_split = tf.split(value=x_data, num_or_size_splits=[input_len])

# outputs: unrolling size x hidden size, state = hidden size
outputs, _states = rnn.static_rnn(lstm, x_split, dtype=tf.float32)

# (optional) softmax layer
softmax_w = tf.get_variable("softmax_w", [input_len, dic_size])
softmax_b = tf.get_variable("softmax_b", [dic_size])
outputs = outputs * softmax_w + softmax_b

outputs = tf.reshape(outputs, [-1, dic_size])
y_data = tf.reshape(y_data, [-1])
weights = tf.ones([input_len * batch_size])

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [outputs], [y_data], [weights])
cost = tf.reduce_mean(loss) / batch_size
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train_op, cost])
    result = sess.run(tf.argmax(outputs, 1))
    print(''.join([char_set[t] for t in result]), l)
