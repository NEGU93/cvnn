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
        with tf.compat.v1.name_scope("dense_layer") as scope:
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
            self._append_graph_structure(shape)  # Append the graph information to the metadata file
            return tf.compat.v1.identity(out, name="y_out"), variables

    # create_mlp_graph compatible
    # restore_graph_from_meta compatible
    # _init_weights compatible
    # All checkpoint methods are compatible

    @staticmethod
    def act_cart_sigmoid(z):
        return tf.keras.activations.sigmoid(z)


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
        train_label_new[i][test_labels[i]] = 1

    # import pdb; pdb.set_trace()

    rvnn = Rvnn(name="RVNN_fashion_MNIST", automatic_restore=False)
    rvnn.create_mlp_graph([(28*28, ''),
                           (128, tf.keras.activations.sigmoid),
                           (10, tf.keras.activations.softmax)]
                          , type_value=np.float32)

    rvnn.train(train_images, train_label_new, test_images, test_label_new)

    predictions = rvnn.predict(test_images)

    import pdb; pdb.set_trace()
