# coding = utf-8
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import types
import pylab

def draw_correct_line():

    x = np.arange(0, 2 * np.pi, 0.01)
    x = x.reshape((len(x), 1))
    y = np.sin(x)

    pylab.plot(x, y, label = 'sin(x)')
    plt.axhline(linewidth = 1, color = 'r')

def get_train_data():

    train_x = np.random.uniform(0.0, 2 * np.pi, (1))
    return train_x, np.sin(train_x)

def inference(input_data):
    with tf.variable_scope('hidden1'):
        weights = tf.get_variable("weight", [1, 16], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("bias", [1, 16], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)

    with tf.variable_scope('hidden2'):
        weights = tf.get_variable("weight", [16, 16], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("bias", [16], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        hidden2 = tf.sigmoid(tf.matmul(hidden1, weights) + biases)

    with tf.variable_scope('hidden3'):
        weights = tf.get_variable("weight", [16, 16], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("bias", [16], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        hidden3 = tf.sigmoid(tf.matmul(hidden2, weights) + biases)

    with tf.variable_scope('output_layer'):
        weights = tf.get_variable("weight", [16, 1], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("bias", [1], tf.float32,
            initializer = tf.random_normal_initializer(0.0, 1))
        output = tf.matmul(hidden3, weights) + biases
    return output

def train():
    learning_rate = 0.01

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    net_out = inference(x)

    loss_op = tf.square(net_out - y)

    opt  =tf.train.GradientDescentOptimizer(learning_rate)
    train_op = opt.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("Start training...")
        for i in range(1000000):
            train_x, train_y = get_train_data()
            sess.run(train_op, feed_dict = {x: train_x, y: train_y})

            if i % 10000 == 0:
                times = int(i / 10000)
                test_x_ndarray = np.arange(0, 2 * np.pi, 0.01)
                test_y_ndarray = np.zeros([len(test_x_ndarray)])
                ind = 0
                for test_x in test_x_ndarray:
                    test_y = sess.run(net_out, feed_dict={x: test_x, y: 1})
                    np.put(test_y_ndarray, ind, test_y)
                    ind += 1
                draw_correct_line()
                pylab.plot(test_x_ndarray, test_y_ndarray, '--', label = str(times) + 'times')
                pylab.show()

if __name__ == "__main__":
    train()