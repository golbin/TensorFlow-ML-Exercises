import tensorflow as tf
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
train_data = np.loadtxt(script_dir + '/train.txt', unpack=True, dtype='float')
x_data = np.transpose(train_data[0:-1])
y_data = np.reshape(train_data[-1:], (4, 1))

print x_data
print y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 5], -1., 1.))
W2 = tf.Variable(tf.random_uniform([5, 5], -1., 1.))
W3 = tf.Variable(tf.random_uniform([5, 4], -1., 1.))
W4 = tf.Variable(tf.random_uniform([4, 1], -1., 1.))

b1 = tf.Variable(tf.random_uniform([5], -1., 1.))
b2 = tf.Variable(tf.random_uniform([5], -1., 1.))
b3 = tf.Variable(tf.random_uniform([4], -1., 1.))
b4 = tf.Variable(tf.random_uniform([1], -1., 1.))

L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)
L4 = tf.nn.relu(tf.matmul(L3, W3) + b3)

hypothesis = tf.sigmoid(tf.matmul(L4, W4) + b4)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

for step in xrange(10000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})

    if step % 200 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W1), sess.run(W2)

correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction], feed_dict={X: x_data, Y: y_data})
print 'accuracy:', sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
# print 'accuracy:', accuracy.eval({X: x_data, Y: y_data})