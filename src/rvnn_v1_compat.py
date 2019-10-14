from cvnn_v1_compat import Cvnn
import data_processing as dp
import tensorflow as tf
import numpy as np
from pdb import set_trace
import sys

"""
# TODO: Is it possible to merge RVNN with CVNN? Check this!
"""


class Rvnn(Cvnn):
    # Init compatible
    # del compatible
    # train compatible (Nicely done Agustin!)
    # predict compatible!
    @staticmethod
    def _create_dense_layer(input_size, output_size, input):
        # TODO: this can actually be a function of cvnn
        # Create weight matrix initialized randomely from N~(0, 0.01)
        w = tf.Variable(np.random.rand(input_size, output_size).astype(np.float32), name="weights")
        b = tf.Variable(np.random.rand(output_size).astype(np.float32), name="bias")

        return tf.add(tf.matmul(input, w), b), [w, b]

    def _create_graph_from_shape(self, shape, type_value=np.float32):
        if len(shape) < 2:
            sys.exit("Cvnn::_create_graph_from_shape: shape should be at least of lenth 2")
        # Define placeholders
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, shape[0][0]], name='X')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, shape[-1][0]], name='Y')

        variables = []
        with tf.compat.v1.name_scope("forward_phase") as scope:
            out = self.apply_activation(shape[0][1], self.X)
            for i in range(len(shape) - 1):  # Apply all the layers
                out, variable = self._create_dense_layer(shape[i][0], shape[i + 1][0], out)
                variables.extend(variable)
                out = self.apply_activation(shape[i + 1][1], out)  # Apply activation function
            return tf.compat.v1.identity(out, name="y_out"), variables

    # create_mlp_graph compatible
    # restore_graph_from_meta compatible
    # _init_weights compatible
    # All checkpoint methods are compatible

    @staticmethod
    def act_cart_sigmoid(z):
        return tf.keras.activations.sigmoid(z)


if __name__ == "__main__":
    m = 100000
    n = 1000
    x = np.ones((m, n))
    w = np.random.rand(n, 1)
    y = np.matmul(x, w)

    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(x, y)

    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    # Network Declaration
    auto_restore = False
    rvnn = Rvnn("RVNN_1HL_for_linear_data", automatic_restore=auto_restore)

    if not auto_restore:
        rvnn.create_mlp_graph([(input_size, 'ignored'), (hidden_size, rvnn.act_cart_sigmoid), (output_size, '')])

    rvnn.train(x_train, y_train, x_test, y_test)
    """y_out = cvnn.predict(x_test)
    if y_out is not None:
        print(y_out[:3])
        print(y_test[:3])"""