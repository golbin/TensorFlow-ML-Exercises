import tensorflow as tf
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
train_data = np.loadtxt(script_dir + '/train.txt', unpack=True, dtype='float')
x_data = np.transpose(train_data[0:-1])
y_data = np.reshape(train_data[-1:], (4, 1))

print x_data
print y_data

X = tf.placeholder(tf.float32, name='x-input')
Y = tf.placeholder(tf.float32, name='y-input')

with tf.name_scope('weight') as scope:
    W1 = tf.Variable(tf.random_uniform([2, 5], -1., 1.), name='weight1')
    W2 = tf.Variable(tf.random_uniform([5, 5], -1., 1.), name='weight2')
    W3 = tf.Variable(tf.random_uniform([5, 4], -1., 1.), name='weight3')
    W4 = tf.Variable(tf.random_uniform([4, 1], -1., 1.), name='weight4')

with tf.name_scope('bias') as scope:
    b1 = tf.Variable(tf.random_uniform([5], -1., 1.), name='bias1')
    b2 = tf.Variable(tf.random_uniform([5], -1., 1.), name='bias2')
    b3 = tf.Variable(tf.random_uniform([4], -1., 1.), name='bias3')
    b4 = tf.Variable(tf.random_uniform([1], -1., 1.), name='bias4')

with tf.name_scope('layer2') as scope:
    L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
with tf.name_scope('layer3') as scope:
    L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)
with tf.name_scope('layer4') as scope:
    L4 = tf.nn.relu(tf.matmul(L3, W3) + b3)

with tf.name_scope('hypothesis') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L4, W4) + b4)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_sum = tf.summary.scalar('cost', cost)

w1_hist_sum = tf.summary.histogram('weight1', W1)
w2_hist_sum = tf.summary.histogram('weight2', W2)
w3_hist_sum = tf.summary.histogram('weight3', W3)
w4_hist_sum = tf.summary.histogram('weight4', W4)

b1_hist_sum = tf.summary.histogram('bias1', b1)
b2_hist_sum = tf.summary.histogram('bias2', b2)
b3_hist_sum = tf.summary.histogram('bias3', b3)
b4_hist_sum = tf.summary.histogram('bias4', b4)

y_hist_sum = tf.summary.histogram('y', Y)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/xor_logs', sess.graph)

    for step in xrange(10000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 200 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y:y_data})
            writer.add_summary(summary, step)
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W1), sess.run(W2)

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction], feed_dict={X: x_data, Y: y_data})
    print 'accuracy:', sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
    # print 'accuracy:', accuracy.eval({X: x_data, Y: y_data})
