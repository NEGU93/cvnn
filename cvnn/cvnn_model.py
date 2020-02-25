import os
import sys
import re
import logging
import threading
import numpy as np
from itertools import count  # To count the number of times fit is called
import tensorflow as tf
import cvnn
import cvnn.layers as layers
import cvnn.data_processing as dp
import cvnn.data_analysis as da
from cvnn.utils import randomize, get_next_batch
from datetime import datetime
from pathlib import Path
from pdb import set_trace
from tensorflow.keras import Model

try:
    import cPickle as pickle
except ImportError:
    import pickle

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class CvnnModel:  # (Model)
    _fit_count = count(0)  # Used to count the number of layers
    """-------------------------
    # Constructor and Stuff
    -------------------------"""

    def __init__(self, name, shape, loss_fun,
                 verbose=True, tensorboard=True, save_model_checkpoints=False, save_csv_checkpoints=True):
        assert not save_model_checkpoints  # TODO: Not working for the moment, sorry!
        pattern = re.compile("^[2-9][0-9]*")
        assert pattern.match(tf.version.VERSION)  # Check TF version is at least 2
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
        self.root_dir = Path(project_path + self.now.strftime("/log/%Y/%m%B/%d%A/run-%Hh%Mm%S/"))
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # Chekpoints
        self.save_csv_checkpoints = save_csv_checkpoints
        self.tensorboard = tensorboard
        self.save_model_checkpoints = save_model_checkpoints
        self.graph_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/train_func"))
        self.graph_writer = tf.summary.create_file_writer(self.graph_writer_logdir)
        if self.tensorboard:  # After this, fit will have no say if using tensorboard or not.
            train_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/train"))
            test_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/test"))
            weigths_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/weights"))
            self.train_summary_writer = tf.summary.create_file_writer(train_writer_logdir)
            self.test_summary_writer = tf.summary.create_file_writer(test_writer_logdir)
            self.weights_summary_writer = tf.summary.create_file_writer(weigths_writer_logdir)

        self._manage_string(self.summary(), verbose, filename=self.name + "_metadata.txt", mode="x")
        self.plotter = da.Plotter(self.root_dir)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        new_shape = []
        for layer in self.shape:
            if isinstance(layer, layers.ComplexDense):
                new_shape.append(layers.ComplexDense(layer.input_size, layer.output_size,
                                                     activation=layer.activation,
                                                     input_dtype=layer.input_dtype,
                                                     output_dtype=layer.output_dtype,
                                                     weight_initializer=layer.weight_initializer,
                                                     bias_initializer=layer.bias_initializer
                                                     ))
            else:
                sys.exit("Layer " + str(layer) + " unknown")
        return CvnnModel(self.name, new_shape, self.loss_fun,
                         verbose=False, tensorboard=self.tensorboard,
                         save_model_checkpoints=False, save_csv_checkpoints=self.save_csv_checkpoints)

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

    def is_complex(self):
        dtype = self.shape[0].get_input_dtype()
        if dtype == np.complex64 or dtype == np.complex128:
            return True
        else:
            return False

    """
        Checkpoints
    """

    def _run_checkpoint(self, x_train, y_train, x_test, y_test,
                        iteration=0, num_tr_iter=0, total_epochs=0,
                        fast_mode=True, verbose=False, save_fit_filename=None):
        if self.tensorboard:  # Save tensorboard data
            self._tensorboard_checkpoint(x_train, y_train, x_test, y_test)
        if self.save_csv_checkpoints:
            self._save_csv(x_train, y_train, 'train')
            if x_test is not None:
                assert y_test is not None
                self._save_csv(x_test, y_test, 'test')
        # Save model weigths
        if self.save_model_checkpoints:
            if x_test is not None:
                assert y_test is not None
                self.save(x_test, y_test)
            else:
                self.save(x_train, y_train)
        # Run output
        if not fast_mode:
            epoch_str = self._get_str_current_epoch(x_train, y_train,
                                                    self.epochs_done, total_epochs,
                                                    iteration, num_tr_iter, x_test, y_test)
            self._manage_string(epoch_str, verbose, save_fit_filename)

    def _save_csv(self, x, y, filename):
        loss = self.evaluate_loss(x, y)
        acc = self.evaluate_accuracy(x, y)
        if not filename.endswith('.csv'):
            filename += '.csv'
        filename = self.root_dir / filename
        # print("Saving to " + str(filename))
        if not os.path.exists(filename):
            file = open(filename, 'x')
            file.write('loss,accuracy\n')
        else:
            file = open(filename, 'a')
        file.write(str(loss) + ',' + str(acc) + '\n')
        file.close()

    def _tensorboard_checkpoint(self, x_train, y_train, x_test, y_test):
        # Save train loss and accuracy
        train_loss, train_accuracy = self.evaluate(x_train, y_train)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=self.epochs_done)
            tf.summary.scalar('accuracy', train_accuracy * 100, step=self.epochs_done)
        # Save test loss and accuracy
        if x_test is not None:
            assert y_test is not None
            test_loss, test_accuracy = self.evaluate(x_test, y_test)
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=self.epochs_done)
                tf.summary.scalar('accuracy', test_accuracy * 100, step=self.epochs_done)
        # Save weights histogram
        for layer in self.shape:
            layer.save_tensorboard_checkpoint(self.weights_summary_writer, self.epochs_done)

    def save(self, x, y):  # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
        # TODO: TypeError: can't pickle _thread._local objects
        # https://github.com/tensorflow/tensorflow/issues/33283
        loss, acc = self.evaluate(x, y)
        checkpoint_root = self.root_dir + "saved_models/"
        if not os.path.exists(checkpoint_root):
            os.makedirs(checkpoint_root)
        save_name = checkpoint_root + "model_checkpoint_epoch" + str(self.epochs_done) + "loss{0:.4f}acc{1:d}".format(
            loss, int(acc * 100))

        # CvnnModel._extractfromlocal(self)
        with open(save_name.replace(".", ",") + ".pickle", "wb") as saver:
            # set_trace()
            pickle.dump(self.__dict__, saver)
            # for lay in self.shape:
            #    pickle.dump(lay.__dict__, saver)
        # CvnnModel._loadtolocal(self)

    """
    @classmethod
    def _extractfromlocal(cls, model):  # extracts attributes from the local thrading container
        model._thread_local = model._thread_local.__dict__
        for attr in model.__dict__.values():
            if '_thread_local' in dir(attr):
                cls._extractfromlocal(attr)

    @classmethod
    def _loadtolocal(cls, model):   # puts attributes back to the local threading container
        aux=threading.local()
        aux.__dict__.update(model._thread_local)
        model._thread_local = aux
        for attr in model.__dict__.values():
            if '_thread_local' in dir(attr):
                cls._loadtolocal(attr)
    """

    @classmethod  # https://stackoverflow.com/a/2709848/5931672
    def loader(cls, f):
        return pickle.load(f)  # TODO: not yet tested

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

    @run_once
    def _start_graph_tensorflow(self):
        # https://github.com/tensorflow/agents/issues/162#issuecomment-512553963
        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        # Prettier option:
        # https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
        tf.summary.trace_on(graph=True, profiler=True)  # https://www.tensorflow.org/tensorboard/graphs

    @run_once
    def _end_graph_tensorflow(self):
        with self.graph_writer.as_default():
            tf.summary.trace_export(name="graph", step=0, profiler_outdir=self.graph_writer_logdir)

    # Add '@tf.function' to accelerate the code by much!
    @tf.function
    def _train_step(self, x_train_batch, y_train_batch, learning_rate):
        with tf.GradientTape() as tape:
            # Forward mode computation
            with tf.name_scope("Forward_Phase") as scope:
                x_called = self.call(x_train_batch)
            # Loss function computation
            with tf.name_scope("Loss") as scope:
                current_loss = self._apply_loss(y_train_batch, x_called)  # Compute loss

        # Calculating gradient
        with tf.name_scope("Gradient") as scope:
            variables = []
            for lay in self.shape:
                variables.extend(lay.trainable_variables)
            gradients = tape.gradient(current_loss, variables)  # Compute gradients
            assert all(g is not None for g in gradients)

        # Backpropagation
        with tf.name_scope("Optimizer") as scope:
            for i, val in enumerate(variables):
                val.assign(val - learning_rate * gradients[i])

    def fit(self, x_train, y_train, x_test=None, y_test=None, learning_rate=0.01, epochs=10, batch_size=100,
            verbose=True, display_freq=100, fast_mode=False, save_to_file=True):
        # TODO: Consider removing fast_mode
        fit_count = next(self._fit_count)  # Know it's own number
        save_fit_filename = None
        if save_to_file:
            save_fit_filename = "fit_" + str(fit_count) + ".txt"
        if np.shape(x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
            self.logger.error("Batch size was bigger than total amount of examples")

        num_tr_iter = int(len(y_train) / batch_size)  # Number of training iterations in each epoch
        self._manage_string("Starting training...\nLearning rate = " + str(learning_rate) + "\n" +
                            "\nEpochs = " + str(epochs) + "\nBatch Size = " + str(batch_size) + "\n" +
                            self._get_str_evaluate(x_train, y_train, x_test, y_test),
                            verbose, save_fit_filename)
        epochs_done = self.epochs_done

        for epoch in range(epochs):
            self.epochs_done += 1
            # Randomly shuffle the training data at the beginning of each epoch
            x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                # Get the batch
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                # Run optimization op (backpropagation)
                if (self.epochs_done * batch_size + iteration) % display_freq == 0:
                    self._run_checkpoint(x_train, y_train, x_test, y_test,
                                         iteration=iteration, num_tr_iter=num_tr_iter,
                                         total_epochs=epochs_done + epochs, fast_mode=fast_mode,
                                         verbose=verbose, save_fit_filename=save_fit_filename)
                self._start_graph_tensorflow()
                self._train_step(x_batch, y_batch, learning_rate)
                self._end_graph_tensorflow()
        # After epochs
        self._run_checkpoint(x_train, y_train, x_test, y_test, fast_mode=True)
        self._manage_string("Train finished...\n" + self._get_str_evaluate(x_train, y_train, x_test, y_test),
                            verbose, save_fit_filename)
        self.plotter.reload_data()

    """
        Managing strings
    """

    def _get_str_current_epoch(self, x, y, epoch, epochs, iteration, num_tr_iter, x_val=None, y_val=None):
        current_loss, current_acc = self.evaluate(x, y)
        ret_str = "Epoch: {0}/{1}; batch {2}/{3}; train loss: " \
                  "{4:.4f} train accuracy: {5:.2f} %".format(epoch, epochs,
                                                             iteration,
                                                             num_tr_iter,
                                                             current_loss,
                                                             current_acc * 100)
        if x_val is not None:
            assert y_val is not None
            val_loss, val_acc = self.evaluate(x_val, y_val)
            ret_str += "; validation loss: {0:.4f} validation accuracy: {1:.2f} %".format(val_loss, val_acc * 100)
        return ret_str + "\n"

    def _manage_string(self, string, verbose=False, filename=None, mode="a"):
        if verbose:
            print(string, end='')
        if filename is not None:
            filename = self.root_dir / filename
            try:
                with open(filename, mode) as file:
                    file.write(string)
            except FileExistsError:  # TODO: Check if this is the actual error
                logging.error("CvnnModel::manage_string: Same file already exists. Aborting to not override results" +
                              str(filename))
            except FileNotFoundError:
                logging.error("CvnnModel::manage_string: No such file or directory: " + self.root_dir)
                sys.exit(-1)

    def _get_str_evaluate(self, x_train, y_train, x_test, y_test):
        loss, acc = self.evaluate(x_train, y_train)
        ret_str = "---------------------------------------------------------\n"
        ret_str += "Training Loss: {0:.4f}, Training Accuracy: {1:.2f} %\n".format(loss, acc * 100)
        if x_test is not None:
            assert y_test is not None
            loss, acc = self.evaluate(x_test, y_test)
            ret_str += "Validation Loss: {0:.4f}, Validation Accuracy: {1:.2f} %\n".format(loss, acc * 100)
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
__version__ = '0.2.13'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
