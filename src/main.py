from activation_functions import *
import tensorflow as tf
from cvnn_v1_compat import Cvnn
from rvnn_v1_compat import Rvnn
import data_processing as dp
import data_analysis as da
import numpy as np


def get_data_1h_for_gauss_noise(x_train, y_train):
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

    return shape_cvnn, shape_rvnn, name


def train_cvnn_and_rvnn(shape_cvnn, shape_rvnn, name):
    # Network Declaration
    auto_restore = False
    cvnn = Cvnn("CVNN" + name, automatic_restore=auto_restore)
    rvnn = Rvnn("RVNN" + name, automatic_restore=auto_restore)

    if not auto_restore:
        # cvnn.create_linear_regression_graph(input_size, output_size)
        cvnn.create_mlp_graph(shape_cvnn, np.float32)

        rvnn.create_mlp_graph(shape_rvnn, np.float32)

    # Train both networks
    cvnn.train(x_train, y_train, x_test, y_test, epochs=10)
    rvnn.train(x_train_real, y_train, x_test_real, y_test, epochs=10)

    return cvnn, rvnn


if __name__ == "__main__":
    # Get data for both networks
    m = 100000      # Number of examples
    n = 1000        # Size of vector
    num_classes = 2
    iterations = 10

    """
    file = open("./results/histogram.csv", 'a')     # TODO: create directory if it does not exist
    file.write("CVNN loss, RVNN loss\n")
    for i in range(iterations):
        x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n, num_classes)
        x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)

        shape_cvnn, shape_rvnn, name = get_data_1h_for_gauss_noise(x_train, y_train)
        cvnn, rvnn = train_cvnn_and_rvnn(shape_cvnn, shape_rvnn, name)

        # compare them
        file.write(str(cvnn.compute_loss(x_test, y_test)))
        file.write(", ")
        file.write(str(rvnn.compute_loss(x_test_real, y_test)))
        file.write("\n")

    file.close()
    """
    da.plot_csv_histogram("./results/histogram.csv")



