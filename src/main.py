from activation_functions import *
import tensorflow as tf
from cvnn_v1_compat import Cvnn
from rvnn_v1_compat import Rvnn
import data_processing as dp
import data_analysis as da
import numpy as np
from pdb import set_trace
import os


def monte_carlo_loss_gaussian_noise(iterations=1000, m=100000, n=1000, num_classes=2, path="./results/",
                                    filename="histogram.csv"):
    """
    Computes the CVNN and RVNN loss result on classification of random gaussian noise over the amount of
    iterations gives as a parameter. Saves results appended to a new csv.
    :return: saves a csv with 2 columns, one for CVNN result and one for RVNN
    """
    if not os.path.exists(path):
        os.makedirs(path)

    write = True
    if os.path.exists(path + filename):
        write = False

    file = open(path + filename, 'a')  # TODO: create directory if it does not exist
    if write:
        file.write("CVNN loss, RVNN loss\n")
    for i in range(iterations):
        x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n, num_classes)
        x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)
        cvnn, rvnn = do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real)

        # compare them
        file.write(str(cvnn.compute_loss(x_test, y_test)))
        file.write(", ")
        file.write(str(rvnn.compute_loss(x_test_real, y_test)))
        file.write("\n")

    file.close()

    da.plot_csv_histogram(path, filename, visualize=True)


def do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real):
    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    shape_cvnn = [(input_size, 'ignored'),
                  (hidden_size, act_cart_sigmoid),
                  (output_size, act_cart_softmax_real)]
    shape_rvnn = [(2 * input_size, 'ignored'),
                  (2 * hidden_size, tf.keras.activations.sigmoid),
                  (output_size, tf.keras.activations.softmax)]
    name = "_1HL_for_gauss_noise"

    auto_restore = False
    cvnn = Cvnn("CVNN" + name, automatic_restore=auto_restore)
    rvnn = Rvnn("RVNN" + name, automatic_restore=auto_restore)

    if not auto_restore:
        # cvnn.create_linear_regression_graph(input_size, output_size)
        cvnn.create_mlp_graph(shape_cvnn, np.float32)
        rvnn.create_mlp_graph(shape_rvnn, np.float32)

    cvnn.train(x_train, y_train, x_test, y_test, epochs=10)
    rvnn.train(x_train_real, y_train, x_test_real, y_test, epochs=10)

    # import pdb; pdb.set_trace()

    return cvnn, rvnn


if __name__ == "__main__":
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 100000
    n = 1000
    num_classes = 2
    x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n, num_classes)
    x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)

    cvnn, rvnn = do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real)

    # set_trace()
