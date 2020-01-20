from cvnn_v1_compat import Cvnn
import data_processing as dp
import tensorflow as tf
import numpy as np
from utils import *
from pdb import set_trace
import sys


"""
THIS MODULE WAS USED TO DEBUG RVNN when using the Cvnn class.
It uses keras fashion dataset and trains a classifier.
"""


def debug_rvnn(x_train, y_train, x_test, y_test):
    tf.compat.v1.disable_eager_execution()
    # Reset latest graph
    tf.compat.v1.reset_default_graph()

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]], name='X')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_test.shape[1]], name='Y')

    with tf.compat.v1.name_scope("input_layer") as scope:
        # Create weight matrix initialized randomely from N~(0, 0.01)
        w = tf.Variable((2*np.random.rand(x_train.shape[1], 128)-1).astype(np.float32), name="weights0")
        b = tf.Variable(np.zeros(128).astype(np.float32), name="bias0")
        h = tf.keras.activations.sigmoid(tf.add(tf.matmul(X, w), b))
    with tf.compat.v1.name_scope("hidden_layer") as scope:
        # Create weight matrix initialized randomely from N~(0, 0.01)
        w1 = tf.Variable((2*np.random.rand(128, y_test.shape[1])-1).astype(np.float32), name="weights1")
        b1 = tf.Variable(np.zeros(y_test.shape[1]).astype(np.float32), name="bias1")
        y_out = tf.keras.activations.softmax(tf.add(tf.matmul(h, w1), b1))
    with tf.compat.v1.name_scope("loss_scope") as scope:
        y1_error = tf.math.multiply(-y, tf.math.log(y_out))  # Error for y = 1
        y0_error = tf.math.multiply(1 - y, tf.math.log(1 - y_out))  # Error for y = 0
        error = tf.math.subtract(y1_error, y0_error)
        loss = tf.reduce_mean(input_tensor=error, name="loss")

    gradients = tf.gradients(ys=loss, xs=[w, b, w1, b1])
    lr_const = 1
    training_op_w = tf.compat.v1.assign(w, w - lr_const * gradients[0])
    training_op_b = tf.compat.v1.assign(b, b - lr_const * gradients[1])
    training_op_w1 = tf.compat.v1.assign(w1, w1 - lr_const * gradients[2])
    training_op_b1 = tf.compat.v1.assign(b1, b1 - lr_const * gradients[3])
    training_op = [training_op_w, training_op_b, training_op_w1, training_op_b1]
    # learning_rate = 1
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # training_op = optimizer.minimize(loss)

    init = tf.compat.v1.global_variables_initializer()

    batch_size = 100
    epochs = 10
    display_freq = 1000
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # print valid
        feed_dict_valid = {X: x_test, y: y_test}
        loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.4f}".format(0, loss_valid))
        print('---------------------------------------------------------')

        num_tr_iter = int(len(y_train) / batch_size)
        for epoch in range(epochs):
            # Randomly shuffle the training data at the beginning of each epoch
            x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                # Run optimization op (backprop)
                feed_dict_batch = {X: x_batch, y: y_batch}
                sess.run(training_op, feed_dict=feed_dict_batch)
                if (epoch * batch_size + iteration) % display_freq == 0:
                    loss_batch = sess.run(loss, feed_dict=feed_dict_batch)
                    print("epoch {0:3d}:\t iteration {1:3d}:\t Loss={2:.2f}".format(epoch, iteration, loss_batch))

        # print valid
        feed_dict_valid = {X: x_test, y: y_test}
        loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.4f}".format(0, loss_valid))
        print('---------------------------------------------------------')
        predictions = sess.run(y_out, feed_dict=feed_dict_valid)
        import pdb; pdb.set_trace()
        print(predictions[0])
        print(np.where(predictions[0] == np.amax(predictions[0])))


if __name__ == "__main__":
    # Get dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images.shape = (train_images.shape[0], train_images.shape[1]*train_images.shape[2])
    test_images.shape = (test_images.shape[0], test_images.shape[1] * test_images.shape[2])

    train_label_new = np.zeros((train_labels.shape[0], 10))
    for i in range(len(train_labels)):
        train_label_new[i][train_labels[i]] = 1
    test_label_new = np.zeros((test_labels.shape[0], 10))
    for i in range(len(test_labels)):
        test_label_new[i][test_labels[i]] = 1

    # debug_rvnn(train_images, train_label_new, test_images, test_label_new)

    rvnn = Cvnn(name="RVNN_fashion_MNIST", automatic_restore=False, learning_rate=1)
    rvnn.create_mlp_graph([(28*28, ''),
                           (128, tf.keras.activations.sigmoid),
                           (10, tf.keras.activations.softmax)],
                          input_dtype=np.float32)

    rvnn.train(train_images, train_label_new, test_images, test_label_new)

    predictions = rvnn.predict(test_images)

    rvnn.compute_accuracy(test_images, test_label_new)

__author__ = 'J. Agustin BARRACIHNA'
__version__ = '1.0.0'
__maintainer__ = 'J. Agustin BARRACIHNA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
