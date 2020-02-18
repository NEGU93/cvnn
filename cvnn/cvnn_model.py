import os
import sys
import logging
import numpy as np
from itertools import count     # To count the number of times fit is called
import tensorflow as tf
import cvnn
import cvnn.layers as layers
import cvnn.data_processing as dp
from cvnn.utils import randomize, get_next_batch
from datetime import datetime
from pdb import set_trace
from tensorflow.keras import Model

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class CvnnModel:  # (Model)
    _fit_count = count(0)  # Used to count the number of layers
    """-------------------------
    # Constructor and Stuff
    -------------------------"""

    def __init__(self, name, shape, loss_fun, verbose=True, tensorboard=True):
        # super(CvnnModel, self).__init__()
        self.name = name
        self.shape = shape
        self.loss_fun = loss_fun
        self.epochs_done = 0
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

        # Folder management for logs
        self.now = datetime.today()
        project_path = os.path.abspath("./")
        self.root_dir = project_path + "/log/" \
                        + str(self.now.year) + "/" \
                        + str(self.now.month) + self.now.strftime("%B") + "/" \
                        + str(self.now.day) + self.now.strftime("%A") \
                        + "/run-" + self.now.time().strftime("%Hh%Mm%S") + "/"
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.tensorboard = tensorboard
        if self.tensorboard:        # After this, fit will have no say if using tensorboard or not.
            train_log_dir = self.root_dir + "tensorboard_logs/train"
            test_log_dir = self.root_dir + "tensorboard_logs/test"
            weigths_log_dit = self.root_dir + "tensorboard_logs/weights"
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            self.weights_summary_writer = tf.summary.create_file_writer(weigths_log_dit)

        self._manage_string(self.summary(), verbose, filename=self.name + "_metadata.txt", mode="x")

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

    """
        Checkpoints
    """
    def _tensorboard_checkpoint(self, x_train, y_train, x_test, y_test):
        # Save train loss and accuracy
        train_loss, train_accuracy = self.evaluate(x_train, y_train)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=self.epochs_done)
            tf.summary.scalar('accuracy', train_accuracy*100, step=self.epochs_done)
        # Save test loss and accuracy
        if x_test is not None:
            assert y_test is not None
            test_loss, test_accuracy = self.evaluate(x_test, y_test)
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=self.epochs_done)
                tf.summary.scalar('accuracy', test_accuracy * 100, step=self.epochs_done)
        # Save weights histogram
        for layer in shape:
            layer.save_tensorboard_checkpoint(self.weights_summary_writer, self.epochs_done)

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
    def _train_step(self, x_train_batch, y_train_batch, learning_rate):
        with tf.GradientTape() as tape:
            current_loss = self._apply_loss(y_train_batch, self.call(x_train_batch))  # Compute loss
        variables = []
        for lay in self.shape:
            variables.extend(lay.trainable_variables)
        gradients = tape.gradient(current_loss, variables)  # Compute gradients
        assert all(g is not None for g in gradients)
        for i, val in enumerate(variables):
            val.assign(val - learning_rate * gradients[i])

    def fit(self, x_train, y_train, x_test=None, y_test=None, learning_rate=0.01, epochs=10, batch_size=100,
            verbose=True, display_freq=100, fast_mode=False, save_to_file=True):
        fit_count = next(self._fit_count)        # Know it's own number
        save_fit_filename = None
        if save_to_file:
            save_fit_filename = "fit_" + str(fit_count) + ".txt"
        if np.shape(x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
            self.logger.error("Batch size was bigger than total amount of examples")

        num_tr_iter = int(len(y_train) / batch_size)  # Number of training iterations in each epoch
        self._manage_string("Starting training...\n" + self._get_str_evaluate(x_train, y_train),
                            verbose, save_fit_filename)
        epochs_done = self.epochs_done

        for epoch in range(epochs):
            if not fast_mode:
                self._tensorboard_checkpoint(x_train, y_train, x_test, y_test)
            self.epochs_done += 1
            # Randomly shuffle the training data at the beginning of each epoch
            x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                # Get the batch
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                # Run optimization op (backpropagation)
                if not fast_mode:
                    if (self.epochs_done * batch_size + iteration) % display_freq == 0:
                        epoch_str = self._get_str_current_epoch(x_batch, y_batch,
                                                                self.epochs_done, epochs_done + epochs,
                                                                iteration, num_tr_iter)
                        self._manage_string(epoch_str, verbose, save_fit_filename)
                self._train_step(x_batch, y_batch, learning_rate)
        # After epochs
        self._manage_string("Train finished...\n" + self._get_str_evaluate(x_train, y_train),
                            verbose, save_fit_filename)

    """
        Managing strings
    """

    def _get_str_current_epoch(self, x, y, epoch, epochs, iteration, num_tr_iter):
        current_loss, current_acc = self.evaluate(x, y)
        return "Epoch: {0}/{1}; batch {2}/{3}; loss: {4:.4f} accuracy: {5:.2f} %\n".format(epoch, epochs, iteration,
                                                                                         num_tr_iter, current_loss,
                                                                                         current_acc * 100)

    def _manage_string(self, string, verbose=False, filename=None, mode="a"):
        if verbose:
            print(string, end='')
        if filename is not None:
            filename = self.root_dir + filename
            try:
                with open(filename, mode) as file:
                    file.write(string)
            except FileExistsError:  # TODO: Check if this is the actual error
                logging.error("CvnnModel::manage_string: Same file already exists. Aborting to not override results" +
                              str(filename))
            except FileNotFoundError:
                logging.error("CvnnModel::manage_string: No such file or directory: " + self.root_dir)
                sys.exit(-1)

    def _get_str_evaluate(self, x_test, y_test):
        loss, acc = self.evaluate(x_test, y_test)
        ret_str = "---------------------------------------------------------\n"
        ret_str += "Loss: {0:.4f}, Accuracy: {1:.2f}\n".format(loss, acc * 100)
        ret_str += "---------------------------------------------------------\n"
        return ret_str

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


if __name__ == '__main__':
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 1000
    n = 100
    num_classes = 5
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
    model.fit(x_train.astype(cdtype), y_train, x_test.astype(cdtype), y_test,
              learning_rate=0.1, batch_size=100, epochs=10)
    model.fit(x_train.astype(cdtype), y_train, x_test.astype(cdtype), y_test,
              learning_rate=0.1, batch_size=100, epochs=10)


# How to comment script header
# https://medium.com/@rukavina.andrei/how-to-write-a-python-script-header-51d3cec13731
__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.2.4'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
