import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data
script_dir = os.path.dirname(os.path.abspath(__file__))
mnist = input_data.read_data_sets(script_dir + "/../mnist/data/", one_hot=True)

# parameters
learning_rate = 0.001
traing_epoch = 15
batch_size = 100
display_step = 1

# input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Xavier initialization
def xavier_init(num_input, num_output, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (num_input + num_output))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (num_input + num_output))
        return tf.truncated_normal_initializer(stddev=stddev)

# weights & bias for nn layers
W1 = tf.get_variable('W1', shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable('W2', shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable('W3', shape=[256, 10], initializer=xavier_init(256, 10))

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

# define layers to build my model
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
hypothesis = tf.add(tf.matmul(L2, W3), b3)

# define cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train my model
for epoch in range(traing_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch

    if epoch % display_step == 0:
        print 'Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost)

print 'Optimization Finished!'

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print 'Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
