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
import cvnn.dataset as dp
import cvnn.data_analysis as da
from cvnn.utils import randomize, get_next_batch, create_folder
from datetime import datetime
import pandas as pd
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


class CvnnModel:
    _fit_count = count(0)  # Used to count the number of layers
    # =====================
    # Constructor and Stuff
    # =====================

    def __init__(self, name, shape, loss_fun,
                 verbose=True, tensorboard=True, save_model_checkpoints=False, save_csv_checkpoints=True):
        """
        Constructor
        :param name: Name of the model. It will be used to distinguish models
        :param shape: List of cvnn.layers.ComplexLayer objects
        :param loss_fun: tensorflow.python.keras.losses to be used.
        :param verbose: if True it will print information of the model just created
        :param tensorboard: If true it will save tensorboard information inside log/.../tensorboard_logs/
                - Loss and accuracy
                - Graph
                - Weights histogram
        :param save_model_checkpoints: Save the model to be able to load and continue training later (Not yet working)
        :param save_csv_checkpoints: Save information of the train and test loss and accuracy on csv files.
        """
        assert not save_model_checkpoints  # TODO: Not working for the moment, sorry!
        pattern = re.compile("^[2-9][0-9]*")
        assert pattern.match(tf.version.VERSION)  # Check TF version is at least 2
        self.name = name
        # Check all the data is a Layer object
        if not all([isinstance(layer, layers.ComplexLayer) for layer in self.shape]):
            self.logger.error("CVNN: All layers in shape must be a cvnn.layer.Layer")
            sys.exit(-1)
        self.shape = shape
        self.loss_fun = loss_fun
        self.epochs_done = 0
        self.run_pandas = pd.DataFrame()
        if not tf.executing_eagerly():      # Make sure nobody disabled eager execution before running the model
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
        self.root_dir = create_folder("./log/", now=self.now)   # Create folder to store all the information

        # Chekpoints
        self.save_csv_checkpoints = save_csv_checkpoints
        self.tensorboard = tensorboard
        self.save_model_checkpoints = save_model_checkpoints
        # The graph will be saved no matter what. It is only computed once so it won't impact much on the training
        self.graph_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/train_func"))
        self.graph_writer = tf.summary.create_file_writer(self.graph_writer_logdir)
        if self.tensorboard:
            train_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/train"))
            test_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/test"))
            weights_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/weights"))
            self.train_summary_writer = tf.summary.create_file_writer(train_writer_logdir)
            self.test_summary_writer = tf.summary.create_file_writer(test_writer_logdir)
            self.weights_summary_writer = tf.summary.create_file_writer(weights_writer_logdir)

        print("Saving {}/{}_metadata.txt".format(self.root_dir, self.name))     # To debug the warning message
        self._manage_string(self.summary(), verbose, filename=self.name + "_metadata.txt", mode="x")
        self.plotter = da.Plotter(self.root_dir)

    def __deepcopy__(self, memodict=None):
        """
        This function is used to create a copy of the model.
        Used for the Monte Carlo simulation. Creates a copy of the model and then trains them.
        ATTENTION: This does not keep the model's weights but randomly initializes them.
            (makes sense like that for the Monte Carlo simulation)
        :param memodict:
        :return: A copy of the current model
        """
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
        """
        Forward result of the network
        :param x: Data input to be calculated
        :return: Output of the netowrk
        """
        for i in range(len(self.shape)):  # Apply all the layers
            x = self.shape[i].call(x)
        return x

    def _apply_loss(self, y_true, y_pred):
        """
        Private! Use "evaluate loss" instead
        """
        # TODO: This can actually be static and give the parameter of loss_fun?
        if callable(self.loss_fun):
            if self.loss_fun.__module__ != 'tensorflow.python.keras.losses':
                self.logger.error("Unknown loss function.\n\t "
                                  "Can only use losses declared on tensorflow.python.keras.losses")
                sys.exit(-1)
        return tf.reduce_mean(input_tensor=self.loss_fun(y_true, y_pred), name=self.loss_fun.__name__)

    def is_complex(self):
        """
        :return: True if the network is complex. False otherwise.
        """
        dtype = self.shape[0].get_input_dtype()
        if dtype == np.complex64 or dtype == np.complex128:
            return True
        else:
            return False

    # ===========
    # Checkpoints
    # ===========

    def _run_checkpoint(self, x_train, y_train, x_test=None, y_test=None,
                        iteration=0, num_tr_iter=0, total_epochs=0,
                        fast_mode=True, verbose=False, save_fit_filename=None):
        """
        Saves whatever needs to be saved (tensorboard, csv of train and test acc and loss, model weigths, etc.
        :param x_train: Train data
        :param y_train: Tran labels
        :param x_test: Test data (optional)
        :param y_test: Test labels (optional)
        :param iteration: Step of the training.
        :param num_tr_iter: Total number of iterations per epoch
        :param total_epochs: Total epochs to be done on the training
        :param fast_mode: Prevents printing results and saving it to the txt. Takes precedence over verbose.
        :param verbose: Print the results on console to visualize the training step. (Unless fast_mode = True)
        :param save_fit_filename: Filename to save the training messages. If None, no information will be saved.
        :return: None
        """
        if self.tensorboard:    # Save tensorboard data
            self._tensorboard_checkpoint(x_train, y_train, x_test, y_test)
        # Save train and (maybe) test acc and loss
        # I do this instead of making and internal vector because I make sure that it can be recovered at any time.
        # Even if the training stopped in the middle by any reason. This way my result is saved many times (Backup!)
        # Other more efficient method would be to create a vector and save it at the end but I risk loosing info
        # if the training stops at any point.
        if self.save_csv_checkpoints:
            self._save_csv(x_train, y_train, 'train')
            if x_test is not None:
                assert y_test is not None, "CVNN::_run_checkpoint: x_test was not None but y_test was None"
                self._save_csv(x_test, y_test, 'test')
        if self.save_model_checkpoints:             # Save model weights
            if x_test is not None:                  # Better to save the loss and acc of test
                assert y_test is not None, "CVNN::_run_checkpoint: x_test was not None but y_test was None"
                self.save(x_test, y_test)
            else:
                self.save(x_train, y_train)         # If I have no info of the test then save the values of the train
        if not fast_mode:       # Print checkpoint state (and maybe save to file)
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
        if not os.path.exists(filename):        # TODO: Can this pose a problem in parallel computing?
            file = open(filename, 'x')
            file.write('loss,accuracy\n')
        else:
            file = open(filename, 'a')
        file.write(str(loss) + ',' + str(acc) + '\n')
        file.close()

    def _tensorboard_checkpoint(self, x_train, y_train, x_test=None, y_test=None):
        """
        Saves the tensorboard data
        :param x_train: Train data
        :param y_train: Tran labels
        :param x_test: Test data (optional)
        :param y_test: Test labels (optional)
        :return: None
        """
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
        os.makedirs(checkpoint_root, exist_ok=True)
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

    # ==========================
    # Predict models and results
    # ==========================

    def predict(self, x):
        """
        Predicts the value of the class.
        ATTENTION: Use this only for classification tasks. For regression use "call" method.
        :param x: Input
        :return: Prediction of the class that x belongs to.
        """
        y_out = self.call(x)
        return tf.math.argmax(y_out, 1)

    def evaluate_loss(self, x, y):
        """
        Computes the output of x and computes the loss using y
        :param x: Input of the netwotk
        :param y: Labels
        :return: loss value
        """
        return self._apply_loss(y, self.call(x)).numpy()

    def evaluate_accuracy(self, x, y):
        """
        Computes the output of x and returns the accuracy using y as labels
        :param x: Input of the netwotk
        :param y: Labels
        :return: accuracy
        """
        y_pred = self.predict(x)
        y_labels = tf.math.argmax(y, 1)
        return tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(y_pred, y_labels), tf.float64)).numpy()

    def evaluate(self, x, y):
        """
        Compues both the loss and accuracy using "evaluate_loss" and "evaluate_accuracy"
        :param x: Input of the netwotk
        :param y: Labels
        :return: tuple (loss, accuracy)
        """
        return self.evaluate_loss(x, y), self.evaluate_accuracy(x, y)

    # ====================
    #          Train 
    # ====================

    @run_once
    def _start_graph_tensorflow(self):
        # https://github.com/tensorflow/agents/issues/162#issuecomment-512553963
        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        # https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
        tf.summary.trace_on(graph=True, profiler=True)  # https://www.tensorflow.org/tensorboard/graphs

    @run_once
    def _end_graph_tensorflow(self):
        with self.graph_writer.as_default():
            tf.summary.trace_export(name="graph", step=0, profiler_outdir=self.graph_writer_logdir)

    # Add '@tf.function' to accelerate the code by a lot!
    @tf.function
    def _train_step(self, x_train_batch, y_train_batch, learning_rate):
        """
        Performs one step of the training
        :param x_train_batch: input
        :param y_train_batch: labels
        :param learning_rate: learning rate fot the gradient descent
        :return: None
        """
        with tf.GradientTape() as tape:
            with tf.name_scope("Forward_Phase") as scope:
                x_called = self.call(x_train_batch)     # Forward mode computation
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
                val.assign(val - learning_rate * gradients[i])      # TODO: For the moment the optimization is only GD

    def fit(self, x_train, y_train, x_test=None, y_test=None, learning_rate=0.01, epochs=10, batch_size=32,
            verbose=True, display_freq=100, fast_mode=False, save_to_file=True):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).
        :param x_train: Input data. # TODO: can use dataset class to make this better
        :param y_train: Labels
        :param x_test: Test data (optional) - Only used for printing results. Will not be use for training any param.
        :param y_test: Test labels (optional)
        :param learning_rate: Learning rate for the gradient descent. For the moment only GD is supported.
        :param epochs: (uint) Number of epochs to do.
        :param batch_size: (uint) Batch size of the data. Default 32 (because keras use 32 so... why not?)
        :param verbose: (Boolean) Print results of the training while training
        :param display_freq: Frequency on terms of steps for saving information and running a checkpoint
        :param fast_mode: (Boolean) Takes precedence over "verbose" and "save_to_file"
        :param save_to_file: (Boolean) save a txt with the information of the fit
                    (same as what will be printed if "verbose")
        :return: None
        """
        assert isinstance(epochs, int) and epochs > 0, "Epochs must be unsigned integer"
        assert isinstance(batch_size, int) and batch_size > 0, "Epochs must be unsigned integer"
        assert learning_rate > 0, "Learning rate must be positive"
        if np.shape(x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
            self.logger.error("Batch size was bigger than total amount of examples")
        # TODO: Consider removing fast_mode
        fit_count = next(self._fit_count)  # Know it's own number. Used to save several fit_<fit_count>.txt
        save_fit_filename = None
        if save_to_file:
            save_fit_filename = "fit_" + str(fit_count) + ".txt"

        num_tr_iter = int(len(y_train) / batch_size)  # Number of training iterations in each epoch
        self._manage_string("Starting training...\nLearning rate = " + str(learning_rate) + "\n" +
                            "\nEpochs = " + str(epochs) + "\nBatch Size = " + str(batch_size) + "\n" +
                            self._get_str_evaluate(x_train, y_train, x_test, y_test),
                            verbose, save_fit_filename)
        epochs_done = self.epochs_done

        for epoch in range(epochs):
            self.epochs_done += 1
            # Randomly shuffle the training data at the beginning of each epoch
            x_train, y_train = randomize(x_train, y_train)      # TODO: keras makes this optional with shuffle opt.
            for iteration in range(num_tr_iter):
                # Get the next batch
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                # Save checkpoint if needed
                if (self.epochs_done * batch_size + iteration) % display_freq == 0:
                    self._run_checkpoint(x_train, y_train, x_test, y_test,
                                         iteration=iteration, num_tr_iter=num_tr_iter,
                                         total_epochs=epochs_done + epochs, fast_mode=fast_mode,
                                         verbose=verbose, save_fit_filename=save_fit_filename)
                # Run optimization op (backpropagation)
                self._start_graph_tensorflow()
                self._train_step(x_batch, y_batch, learning_rate)
                self._end_graph_tensorflow()
        # After epochs
        self._run_checkpoint(x_train, y_train, x_test, y_test, fast_mode=True)
        self._manage_string("Train finished...\n" + self._get_str_evaluate(x_train, y_train, x_test, y_test),
                            verbose, save_fit_filename)
        self.plotter.reload_data()

    # ================
    # Managing strings
    # ================

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
        """
        Prints a string to console and/or saves it to a file
        :param string: String to be printed/saved
        :param verbose: (Boolean) If True it will print the string (default: False)
        :param filename: Filename where to save the string. If None it will not save it (default: None)
        :param mode: Mode to open the filename
        :return: None
        """
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
        # TODO: Use the print function of JG
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
        """
        Generates a string of a summary representation of your model.
        :return: string of the summary of the model
        """
        summary_str = ""
        summary_str += self.name + "\n"
        if self.is_complex():
            summary_str += "Complex Network\n"
        else:
            summary_str += "Real Network\n"
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
__version__ = '0.2.19'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
