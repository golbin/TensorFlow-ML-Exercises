import tensorflow as tf


char_arr = ['h', 'e', 'l', 'o']
# {'o': 3, 'l': 2, 'e': 1, 'h': 0}
char_dic = {w: i for i, w in enumerate(char_arr)}

# [0, 1, 2, 2, 3]
ground_truth = [char_dic[c] for c in 'hello']

# [[1,0,0,0],  # h
#  [0,1,0,0],  # e
#  [0,0,1,0],  # l
#  [0,0,1,0]], # l
x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)

# settings
rnn_size = len(char_dic) # 4
batch_size = 1
output_size = 4
cell_depth = 1

# RNN Model
rnn_cell_single = tf.nn.rnn_cell.BasicRNNCell(
                    num_units=rnn_size,
                    input_size=None)

if cell_depth > 1:
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_single] * cell_depth)
    # [[1, 4], [1, 4]]
    # state = [tf.zeros([batch_size, rnn_cell_single.state_size])
    #             for i in xrange(cell_depth)]
else:
    rnn_cell = rnn_cell_single
    # [1, 4]
    # state = tf.zeros([batch_size, rnn_cell_single.state_size])

# magic!
state = rnn_cell.zero_state(batch_size, tf.float32)

# [[1,0,0,0]] # h
# [[0,1,0,0]] # e
# [[0,0,1,0]] # l
# [[0,0,1,0]] # l
x_split = tf.split(0, len(char_dic), x_data)

# outputs 4 x Tensor(1, 4)
# state = Tensor(1, 4)
outputs, state = tf.nn.rnn(
                    cell=rnn_cell,
                    inputs=x_split,
                    initial_state=state)

# reshape outputs = 4 x [1, 4] -> [1, 16] -> [4, 4]
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])

# [1 2 2 3]
targets = tf.reshape(ground_truth[1:], [-1])

# [1. 1. 1. 1]
weights = tf.ones([len(char_dic) * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50):
    sess.run(train_op)
    result = sess.run(tf.argmax(logits, 1))
    print result, [char_arr[t] for t in result]
