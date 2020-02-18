import os
import sys
import logging
import numpy as np
import tensorflow as tf
import cvnn
import cvnn.layers as layers
import cvnn.data_processing as dp
from cvnn.utils import randomize, get_next_batch
from datetime import datetime
from pdb import set_trace
from tensorflow.keras import Model

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class OutputOpts:
    def __init__(self, tensorboard, verbose, display_freq=100):
        self.tensorboard = tensorboard
        self.verbose = verbose
        self.display_freq = display_freq


class CvnnModel:  # (Model)
    """-------------------------
    # Constructor and Stuff
    -------------------------"""

    def __init__(self, name, shape, loss_fun, verbose=True):
        # super(CvnnModel, self).__init__()
        self.name = name
        self.shape = shape
        self.loss_fun = loss_fun
        if verbose:
            self.print_summary()
        if not tf.executing_eagerly():
            # tf.compat.v1.enable_eager_execution()
            logging.error("CvnnModel::__init__: TF was not executing eagerly")
            sys.exit(-1)

        # Logging parameters
        logging.getLogger('tensorflow').disabled = True
        logger = logging.getLogger(cvnn.__name__)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        self.logger = logger

        # Folder for saving information
        self.now = datetime.today()
        project_path = os.path.abspath("./")
        self.root_dir = project_path + "/log/" \
                        + str(self.now.year) + "/" \
                        + str(self.now.month) + self.now.strftime("%B") + "/" \
                        + str(self.now.day) + self.now.strftime("%A") \
                        + "/run-" + self.now.time().strftime("%Hh%Mm%S") + "/"
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.save_metadata()

    def call(self, x):
        # Check all the data is a Layer object
        if not all([isinstance(layer, layers.ComplexLayer) for layer in self.shape]):
            self.logger.error("CVNN::_create_graph_from_shape: all layers in shape must be a cvnn.layer.Layer")
            sys.exit(-1)
        for i in range(len(self.shape)):  # Apply all the layers
            x = self.shape[i].call(x)
        return x

    def _apply_loss(self, y_true, y_pred):
        # TODO: don't like the fact that I have to give self to this and not to apply_activation
        if callable(self.loss_fun):
            if self.loss_fun.__module__ != 'tensorflow.python.keras.losses':
                self.logger.error("Unknown loss function.\n\t "
                                  "Can only use losses declared on tensorflow.python.keras.losses")
        return tf.reduce_mean(input_tensor=self.loss_fun(y_true, y_pred), name=self.loss_fun.__name__)

    """-------------------------
    # Predict models and results
    -------------------------"""

    def predict(self, x):
        y_out = self.call(x)
        return tf.math.argmax(y_out, 1)

    def evaluate_loss(self, x_test, y_true):
        return self._apply_loss(y_true, self.call(x_test)).numpy()

    def evaluate_accuracy(self, x_test, y_true):
        y_pred = self.predict(x_test)
        y_labels = tf.math.argmax(y_true, 1)
        return tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(y_pred, y_labels), tf.float64)).numpy()

    def evaluate(self, x_test, y_true):
        return self.evaluate_loss(x_test, y_true), self.evaluate_accuracy(x_test, y_true)

    """-----------------------
    #          Train 
    -----------------------"""

    # Add '@tf.function' to accelerate the code by much!
    @tf.function
    def train_step(self, x_train_batch, y_train_batch, learning_rate):
        with tf.GradientTape() as tape:
            current_loss = self._apply_loss(y_train_batch, self.call(x_train_batch))  # Compute loss
        variables = []
        for lay in self.shape:
            variables.extend(lay.trainable_variables)
        gradients = tape.gradient(current_loss, variables)  # Compute gradients
        assert all(g is not None for g in gradients)
        for i, val in enumerate(variables):
            val.assign(val - learning_rate * gradients[i])

    def fit(self, x_train, y_train, learning_rate=0.01, epochs=10, batch_size=100,
            verbose=True, display_freq=100):
        if np.shape(x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
            self.logger.error("Batch size was bigger than total amount of examples")

        num_tr_iter = int(len(y_train) / batch_size)  # Number of training iterations in each epoch
        if verbose:
            print("Starting training...")
        for epoch in range(epochs):
            # Randomly shuffle the training data at the beginning of each epoch
            x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                # Get the batch
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                # Run optimization op (backpropagation)
                if verbose and (epoch * batch_size + iteration) % display_freq == 0:
                    current_loss, current_acc = self.evaluate(x_batch, y_batch)
                    print("Epoch: {0}/{1}; batch {2}/{3}; loss: {4:.4f} accuracy: {5:.2f} %"
                          .format(epoch, epochs, iteration, num_tr_iter, current_loss, current_acc * 100))
                self.train_step(x_batch, y_batch, learning_rate)

    """
        Saving and Summary
    """

    def summary(self):
        summary_str = ""
        summary_str += self.name + "\n"
        net_dtype = self.shape[0].get_input_dtype()
        if net_dtype == np.complex64 or net_dtype == np.complex128:
            summary_str += "Complex Network\n"
        elif net_dtype == np.float32 or net_dtype == np.float64:
            summary_str += "Real Network\n"
        else:
            summary_str += "Unknown Data Type Network\n"
            logging.warning("CvnnModel::summary: Unknown Data Type Network")
        for lay in self.shape:
            summary_str += lay.get_description()
        return summary_str

    def print_summary(self):
        print(self.summary())

    def save_metadata(self):
        filename = self.root_dir + self.name + "_metadata.txt"
        try:
            with open(filename, "x") as file:
                file.write(self.summary())
        except FileExistsError:  # TODO: Check if this is the actual error
            logging.error("CvnnModel::save_metadata: Same file already exists. Aborting to not override results")
            sys.exit(-1)
        except FileNotFoundError:
            logging.error("CvnnModel::save_metadata: No such file or directory: " + self.root_dir)
            sys.exit(-1)


if __name__ == '__main__':
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 1000
    n = 100
    num_classes = 2
    x_train, y_train, x_test, y_test = dp.get_gaussian_noise(m, n, num_classes, 'hilbert')
    cdtype = np.complex64
    if cdtype == np.complex64:
        rdtype = np.float32
    else:
        rdtype = np.float64

    x_train = x_train.astype(np.complex64)
    x_test = x_test.astype(np.complex64)

    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    shape = [layers.ComplexDense(input_size=input_size, output_size=hidden_size, activation='cart_sigmoid',
                                 input_dtype=cdtype, output_dtype=cdtype),
             layers.ComplexDense(input_size=hidden_size, output_size=output_size, activation='cart_softmax_real',
                                 input_dtype=cdtype, output_dtype=rdtype)]
    model = CvnnModel("Testing v2 class", shape, tf.keras.losses.categorical_crossentropy)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    print(model.evaluate(x_test.astype(cdtype), y_test, ))
    model.fit(x_train.astype(cdtype), y_train, learning_rate=0.1, batch_size=100, epochs=10)
    print(model.evaluate(x_test.astype(cdtype), y_test, ))


# How to comment script header
# https://medium.com/@rukavina.andrei/how-to-write-a-python-script-header-51d3cec13731
__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.2.1'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
