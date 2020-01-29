from activation_functions import *
import tensorflow as tf
from cvnn_v1_compat import Cvnn
import data_processing as dp
import numpy as np
from pdb import set_trace
import sys
import os


def monte_carlo_cvnn_rvnn_compare(iterations=100, m=10000, n=100, min_num_classes=2, max_num_classes=5,
                                  test_for_each_data=10, path="../results/", filename="histogram", name=''):
    """
    Computes the CVNN and RVNN loss and acc result on classification of random gaussian noise over the amount of
    iterations gives as a parameter. Saves results appended to a new csv.
    It goes from 2 classes to num_classes for iteration times each.
    Saves a csv with 2 columns, one for CVNN result and one for RVNN.
    :return: None
    """
    if min_num_classes < 2:
        sys.exit("Classes should be at least 2. Got " + str(num_classes))
    elif min_num_classes > max_num_classes:
        sys.exit("min_num_classes (" + str(min_num_classes) + ") should be lower or equal to max_num_classes ("
                 + str(max_num_classes) + ")")
    if not os.path.exists(path):
        os.makedirs(path)

    for k in range(min_num_classes, max_num_classes+1):
        for i in range(iterations):
            print("Iteration " + str(i) + " for " + str(k) + " classes")
            x_train, y_train, x_test, y_test = dp.get_gaussian_noise(m, n, k, name)
            x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)
            write = True
            if os.path.exists(path + filename + "_iter" + str(i) + "_classes" + str(k) + '.csv'):
                write = False   # Not to write again the CVNN and all the headers.
            file = open(path + filename + "_iter" + str(i) + "_classes" + str(k) + '.csv', 'a')
            if write:
                file.write("CVNN loss, CVNN acc, RVNN loss, RVNN acc\n")
            for t in range(test_for_each_data):
                print("Iteration " + str(i) + "." + str(t) + " for " + str(k) + " classes")
                cvnn, rvnn = do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real, verbose=False)

                # compare them
                file.write(str(cvnn.compute_loss(x_test, y_test)))
                file.write(", ")
                file.write(str(cvnn.compute_accuracy(x_test, y_test)))
                file.write(", ")
                file.write(str(rvnn.compute_loss(x_test_real, y_test)))
                file.write(", ")
                file.write(str(rvnn.compute_accuracy(x_test_real, y_test)))
                file.write("\n")
            file.close()

    # da.plot_csv_histogram(path, filename, visualize=True)
    print("Monte carlo finished.")


def do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real, name='', verbose=True):
    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    shape_cvnn = [(input_size, 'ignored'),
                  (hidden_size, 'cart_sigmoid'),
                  (output_size, 'cart_softmax_real')]
    shape_rvnn = [(2 * input_size, 'ignored'),
                  (2 * hidden_size, tf.keras.activations.sigmoid),
                  (output_size, tf.keras.activations.softmax)]
    name = "_1HL_for_" + name + "_noise"

    auto_restore = False
    cvnn = Cvnn("CVNN" + name, automatic_restore=auto_restore, verbose=verbose)
    rvnn = Cvnn("RVNN" + name, automatic_restore=auto_restore, verbose=verbose)

    if not auto_restore:
        # cvnn.create_linear_regression_graph(input_size, output_size)
        cvnn.create_mlp_graph(tf.keras.losses.categorical_crossentropy, shape_cvnn, input_dtype=np.complex64)
        rvnn.create_mlp_graph(tf.keras.losses.categorical_crossentropy, shape_rvnn, input_dtype=np.float32)

    cvnn.train(x_train, y_train, x_test, y_test, epochs=10)
    rvnn.train(x_train_real, y_train, x_test_real, y_test, epochs=10)

    # import pdb; pdb.set_trace()

    return cvnn, rvnn


if __name__ == "__main__":
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 100000
    n = 100
    num_classes = 4
    name = 'hilbert'
    # random
    # vect = None
    # work
    # vect = [42, 51, 37]
    # vect = [42, 51, 37]
    # vect = [42, 51, 37]
    # vect = [42, 51, 37]
    # Doesn't work
    # vect = [42, 51, 37]

    monte_carlo_cvnn_rvnn_compare(iterations=50, m=m, n=n, min_num_classes=num_classes, max_num_classes=6, name=name)

    """
    x_train, y_train, x_test, y_test = dp.get_gaussian_noise(m, n, num_classes, name)
    # x_train, y_train, x_test, y_test = dp.get_constant(m, n, num_classes, vect)
    x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)

    cvnn, rvnn = do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real, name)

    print(da.categorical_confusion_matrix(cvnn.predict(x_test), y_test))
    print(da.categorical_confusion_matrix(rvnn.predict(x_test_real), y_test))

    set_trace()
    """

__author__ = 'J. Agustin BARRACHINA'
__version__ = '1.0.6'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
