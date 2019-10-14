from activation_functions import *
import tensorflow as tf
from cvnn_v1_compat import Cvnn
from rvnn_v1_compat import Rvnn
import data_processing as dp
import numpy as np


if __name__ == "__main__":
    # Get data for both networks
    m = 100000      # Number of examples
    n = 1000        # Size of vector
    num_classes = 2
    x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n)
    x_train_real = dp.transform_to_real(x_train)
    x_test_real = dp.transform_to_real(x_test)

    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    # Network Declaration
    auto_restore = False
    cvnn = Cvnn("CVNN_1HL_for_gauss_noise", automatic_restore=auto_restore)
    rvnn = Rvnn("RVNN_1HL_for_gauss_noise", automatic_restore=auto_restore)

    if not auto_restore:
        # cvnn.create_linear_regression_graph(input_size, output_size)
        cvnn.create_mlp_graph([(input_size, 'ignored'),
                               (hidden_size, act_cart_sigmoid),
                               (output_size, act_cart_softmax)])

        rvnn.create_mlp_graph([(2*input_size, 'ignored'),
                               (2*hidden_size, tf.keras.activations.sigmoid),
                               (output_size, tf.keras.activations.softmax)])

    # Train both networks
    cvnn.train(x_train, y_train, x_test, y_test)
    rvnn.train(x_train_real, y_train, x_test_real, y_test)

    # compare them
    print("CVNN loss: " + str(cvnn.compute_loss(x_test, y_test)) + "%")
    print("RVNN loss: " + str(rvnn.compute_loss(x_test_real, y_test)) + "%")

