import os
import sys
import copy
import re
import logging
import numpy as np
import pandas as pd
from itertools import count  # To count the number of times fit is called
import tensorflow as tf
from datetime import datetime
from pdb import set_trace
from time import strftime, perf_counter, gmtime
from prettytable import PrettyTable
# My own module!
import cvnn
import cvnn.layers as layers
import cvnn.dataset as dp
import cvnn.data_analysis as da
from cvnn.utils import randomize, create_folder, transform_to_real

try:
    import cPickle as pickle
except ImportError:
    import pickle


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


logger = logging.getLogger(cvnn.__name__)

"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)"""


class CvnnModel:
    _fit_count = count(0)  # Used to count the number of layers

    # =====================
    # Constructor and Stuff
    # =====================

    def __init__(self, name, shape, loss_fun, verbose=True, tensorboard=True):
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
        """
        self.name = name
        # Check all the data is a Layer object
        if not all([isinstance(layer, layers.ComplexLayer) for layer in shape]):
            logger.error("All layers in shape must be a cvnn.layer.Layer", exc_info=True)
            sys.exit(-1)
        self.shape = shape
        self.loss_fun = loss_fun
        self.epochs_done = 0
        self.run_pandas = pd.DataFrame(columns=['step', 'epoch',
                                                'train loss', 'train accuracy', 'test loss', 'test accuracy'])
        if not tf.executing_eagerly():  # Make sure nobody disabled eager execution before running the model
            logging.error("CvnnModel::__init__: TF was not executing eagerly", exc_info=True)
            sys.exit(-1)

        # Folder management for logs
        self.now = datetime.today()
        self.root_dir = create_folder("./log/models/", now=self.now)  # Create folder to store all the information

        # Chekpoints
        self.tensorboard = tensorboard
        # The graph will be saved no matter what. It is only computed once so it won't impact much on the training
        self.graph_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/graph"))
        self.graph_writer = tf.summary.create_file_writer(self.graph_writer_logdir)
        if self.tensorboard:
            train_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/train"))
            test_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/test"))
            weights_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/weights"))
            activation_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/activation"))
            gradients_writer_logdir = str(self.root_dir.joinpath("tensorboard_logs/gradients"))
            self.train_summary_writer = tf.summary.create_file_writer(train_writer_logdir)
            self.test_summary_writer = tf.summary.create_file_writer(test_writer_logdir)
            self.weights_summary_writer = tf.summary.create_file_writer(weights_writer_logdir)
            self.activation_summary_writer = tf.summary.create_file_writer(activation_writer_logdir)
            self.gradients_summary_writer = tf.summary.create_file_writer(gradients_writer_logdir)

        # print("Saving {}/{}_metadata.txt".format(self.root_dir, self.name))     # To debug the warning message
        self._manage_string(self.summary(), verbose, filename=self.name + "_metadata.txt", mode="x")
        self.plotter = da.Plotter(self.root_dir)

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
                logger.error("Unknown loss function.\n\t "
                             "Can only use losses declared on tensorflow.python.keras.losses", exc_info=True)
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

    # ====================
    #     Copy methods
    # ====================

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
        layers.ComplexLayer.last_layer_output_dtype = None
        layers.ComplexLayer.last_layer_output_size = None
        for layer in self.shape:
            if isinstance(layer, layers.ComplexLayer):
                new_shape.append(copy.deepcopy(layer))
            else:
                logger.error("Layer " + str(layer) + " not child of cvnn.layers.ComplexLayer")
                sys.exit(-1)
        return CvnnModel(self.name, new_shape, self.loss_fun, verbose=False, tensorboard=self.tensorboard)

    def _get_real_equivalent_multiplier(self, classifier=True, capacity_equivalent=True):
        if capacity_equivalent and not classifier:  # TODO: Support it, it should not be so difficult
            logger.error("The current code does not support capacity equivalence for logistic regression tasks")
            sys.exit(-1)    # TODO: Make one to be prevalent over the other one and just show a warning
        if capacity_equivalent:
            if len(self.shape) == 1:        # Case with no hidden layers
                output_mult = np.ones(1).astype(int)
            elif len(self.shape) == 2:      # Case only one hidden layer
                n = self.shape[0].input_size
                c = self.shape[-1].output_size
                hidden_1_size = np.ceil(((2 * n + 2 * c) / (2 * n + c))).astype(int)
                output_mult = np.array([hidden_1_size, 1])
            elif len(self.shape) % 2 == 1:      # Case with even hidden layers (odd with the output layer)
                mask = np.ones(len(self.shape)).astype(int)
                mask[::2].fill(0)
                output_mult = np.ones(len(self.shape)).astype(int) + mask
            else:                               # Case with odd hidden layers
                mask = np.ones(len(self.shape)).astype(int)
                mask[::2].fill(0)
                output_mult = np.ones(len(self.shape)).astype(int) + mask
                middle_index = int(len(output_mult)/2 - 1)
                m_inf = self.shape[middle_index-1].output_size
                m_sup = self.shape[middle_index+1].output_size
                value = np.ceil(2*(m_inf + m_sup)/(m_inf + 2*m_sup)).astype(int)
                output_mult = np.insert(output_mult[:-1], middle_index, value)
        else:
            output_mult = 2*np.ones(len(self.shape)).astype(int)
            if classifier:
                output_mult[-1] = 1
        return output_mult

    def get_real_equivalent(self, classifier=True, capacity_equivalent=True, name=None):
        """
        Creates a new model equivalent of current model. If model is already real throws and error.
        :param classifier: True (default) if the model is a classification model. False otherwise.
        :param name: name of the new network to be created.
            If None (Default) it will use same name as current model with "_real_equiv" suffix
        :param capacity_equivalent: If true, it creates a capacity-equivalent model (https://arxiv.org/abs/1811.12351)
            If false, it will double all layers except from the last one.
        :return: CvnnModel() real equivalent model
        """
        if not self.is_complex():
            logger.error("model {} was already real".format(self.name))
            sys.exit(-1)
        # assert len(self.shape) != 0
        real_shape = []
        layers.ComplexLayer.last_layer_output_dtype = None
        layers.ComplexLayer.last_layer_output_size = None
        output_mult = self._get_real_equivalent_multiplier(classifier, capacity_equivalent)
        for i, layer in enumerate(self.shape):
            if isinstance(layer, layers.ComplexLayer):
                if isinstance(layer, layers.Dense):  # TODO: Check if I can do this with kargs or sth
                    real_shape.append(layer.get_real_equivalent(output_multiplier=output_mult[i],
                                                                input_multiplier=output_mult[i-1] if i > 0 else 2))
                else:
                    real_shape.append(layer.get_real_equivalent())
            else:
                sys.exit("Layer " + str(layer) + " unknown")
        if name is None:
            name = self.name + "_real_equiv"
        # set_trace()
        return CvnnModel(name=name, shape=real_shape, loss_fun=self.loss_fun,
                         tensorboard=self.tensorboard, verbose=False)

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
                x_called = self.call(x_train_batch)  # Forward mode computation
            # Loss function computation
            with tf.name_scope("Loss") as scope:
                current_loss = self._apply_loss(y_train_batch, x_called)  # Compute loss

        # Calculating gradient
        with tf.name_scope("Gradient") as scope:
            variables = []
            for lay in self.shape:
                variables.extend(lay.trainable_variables())  # TODO: Debug this for all layers.
            gradients = tape.gradient(current_loss, variables)  # Compute gradients
            assert all(g is not None for g in gradients)

        # Backpropagation
        with tf.name_scope("Optimizer") as scope:
            for i, val in enumerate(variables):
                val.assign(val - learning_rate * gradients[i])  # TODO: For the moment the optimization is only GD

    def fit(self, x, y,
            validation_split=0.0, x_test=None, y_test=None,
            learning_rate=0.01, epochs: int = 10, batch_size: int = 32,
            verbose=True, display_freq=None, fast_mode=True, save_txt_fit_summary=False,
            save_model_checkpoints=False, save_csv_history=True, shuffle=True):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        :param x: Input data.
        :param y: Labels
        :param validation_split: Percentage of the input data to be used as train set (the rest will be use as validation set)
            Default: 0.8 (80% as train set and 20% as validation set)
        :param learning_rate: Learning rate for the gradient descent. For the moment only GD is supported.
        :param epochs: (uint) Number of epochs to do.
        :param batch_size: (uint) Batch size of the data. Default 32 (because keras use 32 so... why not?)
        :param verbose: (Boolean) Print results of the training while training
        :param display_freq: Frequency on terms of steps for saving information and running a checkpoint.
            If None (default) it will automatically match 1 epoch = 1 step (print/save information at each epoch)
        :param fast_mode: (Boolean) Does 2 things if False:
                    1. Saves csv files with each checkpoint
                    2. Prints loss and accuracy if verbose = True
        :param save_txt_fit_summary: (Boolean) save a txt with the information of the fit
                    (same as what will be printed if "verbose")
        :param save_model_checkpoints: Save the model to be able to load and continue training later (Not yet working)
        :param save_csv_history: Save information of the train and test loss and accuracy on csv files.
        :param shuffle: (Boolean) Whether to shuffle the training data before each epoch. Default: True
        :return: None
        """
        # Check input
        assert not save_model_checkpoints  # TODO: Not working for the moment, sorry!
        if not (isinstance(epochs, int) and epochs > 0):
            logger.error("Epochs must be unsigned integer", exc_info=True)
            sys.exit(-1)
        if not (isinstance(batch_size, int) and batch_size > 0):
            logger.error("Batch size must be unsigned integer", exc_info=True)
            sys.exit(-1)
        if not learning_rate > 0:
            logger.error("Learning rate must be positive", exc_info=True)
            sys.exit(-1)
        if display_freq is None:
            display_freq = int((x.shape[0] * (1 - validation_split)) / batch_size)  # Match the epoch number

        # Prepare dataset
        # categorical = (len(np.shape(y)) > 1)
        # dataset = dp.Dataset(x, y, ratio=ratio, batch_size=batch_size, savedata=False, categorical=categorical)
        if x_test is None or y_test is None:
            assert 0 <= validation_split < 1, "Ratio should be between [0, 1)"
            dataset_length = np.shape(x)[0]
            x_train = x[int(dataset_length * validation_split):]
            y_train = y[int(dataset_length * validation_split):]
            x_test = x[:int(dataset_length * validation_split)]
            y_test = y[:int(dataset_length * validation_split)]
            if len(x_test) == 0:
                x_test = None
                y_test = None
        else:
            x_train = x
            y_train = y
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=batch_size)
        """if validation_split != 1:
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        else:
            test_dataset = None"""

        # Create fit txt if needed
        fit_count = next(self._fit_count)  # Know it's own number. Used to save several fit_<fit_count>.txt
        save_fit_filename = None
        if save_txt_fit_summary:
            save_fit_filename = "fit_" + str(fit_count) + ".txt"
        # Print start condition
        start_status = self._get_str_evaluate(self.epochs_done, epochs, x_train, y_train, x_test, y_test,
                                              fast_mode=fast_mode)
        self._manage_string("Starting training...\nLearning rate = " + str(learning_rate) + "\n" +
                            "Epochs = " + str(epochs) + "\nBatch Size = " + str(batch_size) + "\n" +
                            start_status, verbose, save_fit_filename)
        # -----------------------------------------------------
        # input processing ended
        num_tr_iter = int(x_train.shape[0] / batch_size)  # Number of training iterations in each epoch
        epochs_before_fit = self.epochs_done
        start_time = perf_counter()
        for epoch in range(epochs):
            iteration = 0
            if verbose:
                tf.print("\nEpoch {0}/{1}".format(self.epochs_done, epochs))
                progbar = tf.keras.utils.Progbar(num_tr_iter)
            # Randomly shuffle the training data at the beginning of each epoch
            if shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=5000)
            for x_batch, y_batch in train_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache():
                if verbose:
                    progbar.update(iteration)
                iteration += 1
                # Save checkpoint if needed
                if ((epochs_before_fit + epoch) * num_tr_iter + iteration) % display_freq == 0:
                    self._run_checkpoint(x_batch, y_batch, x_test, y_test,    # Shall I use batch to be more efficient?
                                         step=(epochs_before_fit + epoch) * num_tr_iter + iteration,
                                         num_tr_iter=num_tr_iter, total_epochs=epochs_before_fit + epochs,
                                         fast_mode=fast_mode, verbose=False, save_fit_filename=save_fit_filename,
                                         save_model_checkpoints=save_model_checkpoints,
                                         save_csv_checkpoints=save_csv_history)
                # Run optimization op (backpropagation)
                # x_batch, y_batch = dataset.get_next_batch()  # Get the next batch
                self._start_graph_tensorflow()
                self._train_step(x_batch, y_batch, learning_rate)
                self._end_graph_tensorflow()
            self.epochs_done += 1

        # After epochs
        end_time = perf_counter()
        self._run_checkpoint(x_train, y_train, x_test, y_test,
                             step=epochs * num_tr_iter + num_tr_iter, num_tr_iter=num_tr_iter,
                             total_epochs=epochs_before_fit + epochs, fast_mode=False)
        end_status = self._get_str_evaluate(epochs, epochs, x_train, y_train, x_test, y_test, fast_mode=fast_mode)
        self._manage_string("Train finished...\n" +
                            end_status +
                            "\nTraining time: {}s".format(strftime("%H:%M:%S", gmtime(end_time - start_time))),
                            verbose, save_fit_filename)
        self.plotter.reload_data()

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

    def evaluate_train_and_test(self, train_x, train_y, test_x, test_y):
        train_loss, train_acc = self.evaluate(train_x, train_y)
        if test_x is None or test_y is None:
            return train_loss, train_acc, None, None
        else:
            test_loss, test_acc = self.evaluate(test_x, test_y)
            return train_loss, train_acc, test_loss, test_acc

    def get_confusion_matrix(self, x, y, save_result=False):
        """
        Generates a pandas data-frame with the confusion matrix of result of x and y (labels)
        :param x: data to which apply the model
        :param y: labels
        :param save_result: if True it will save the confusion matrix as a csv at models path
        :return: Confusion matrix pandas data-frame
        """
        filename = None
        if save_result:
            filename = self.root_dir / "categorical.csv"
        return da.confusion_matrix(self.call(x), y, filename=filename)

    # ===========
    # Checkpoints
    # ===========

    def _run_checkpoint(self, x_train, y_train, x_test, y_test,
                        step=0, num_tr_iter=0, total_epochs=0,
                        fast_mode=False, verbose=False, save_fit_filename=None,
                        save_model_checkpoints=False, save_csv_checkpoints=True):
        """
        Saves whatever needs to be saved (tensorboard, csv of train and test acc and loss, model weigths, etc.

        :param dataset: dataset object of cvnn.dataset
        :param step: step of the training.
        :param num_tr_iter: Total number of iterations per epoch.
        :param total_epochs: Total epochs to be done on the training.
        :param fast_mode: if True it will save the loss and accuracy as a csv on each file.
                This will prevent the loss of data but will make the training longer.
                Default: False.
        :param verbose: Print the results on console to visualize the training step.
        :param save_fit_filename: Filename to save the training messages. If None, no information will be saved.
        :param save_model_checkpoints: Save the model to be able to load and continue training later (Not yet working)
        :param save_csv_checkpoints: Save information of the train and test loss and accuracy on csv files.
        :return: None
        """
        # First I check if at least one is needed. If not better don't compute the information.
        if save_csv_checkpoints or save_model_checkpoints or verbose or save_fit_filename is not None:
            train_loss, train_acc, test_loss, test_acc = self.evaluate_train_and_test(x_train, y_train, x_test, y_test)

        if self.tensorboard:  # Save tensorboard data
            self._tensorboard_checkpoint(x_train, y_train, x_test, y_test)
        if save_csv_checkpoints:
            # With fast mode False I save the checkpoint in a csv.
            # It will take longer to run because I create a file each time
            # but if I don't do it and something happens I will loose all the information
            self._save_current_loss_and_acc(self.name + '_results_fit',
                                            train_loss, train_acc, test_loss, test_acc, step, fast_mode)
        if save_model_checkpoints:  # Save model weights
            self.save(test_loss, test_acc)
        if save_fit_filename is not None or verbose:  # I first check if it makes sense to get the string
            epoch_str = self._get_loss_and_acc_string(epoch=self.epochs_done, epochs=total_epochs,
                                                      train_loss=train_loss, train_acc=train_acc,
                                                      test_loss=test_loss, test_acc=test_acc,
                                                      batch=step % num_tr_iter, batches=num_tr_iter,
                                                      fast_mode=fast_mode)
            self._manage_string(epoch_str, verbose, save_fit_filename)

    def _save_current_loss_and_acc(self, filename, train_loss, train_acc, test_loss, test_acc, step, fast_mode=False):
        a_series = pd.Series([step, self.epochs_done, train_loss, train_acc, test_loss, test_acc],
                             index=self.run_pandas.columns)
        self.run_pandas = self.run_pandas.append(a_series, ignore_index=True)
        if not filename.endswith('.csv'):
            filename += '.csv'
        filename = self.root_dir / filename
        if not fast_mode:
            self.run_pandas.to_csv(filename, index=False)

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
        x = x_train
        for layer in self.shape:
            x = layer.save_tensorboard_checkpoint(x, self.weights_summary_writer,
                                                  self.activation_summary_writer, self.epochs_done)
        self._save_tensorboard_gradients(x_train, y_train)

    def _save_tensorboard_gradients(self, x_train, y_train):
        with tf.GradientTape() as tape:
            x_called = self.call(x_train)  # Forward mode computation
            current_loss = self._apply_loss(y_train, x_called)  # Compute loss
        variables = []
        for lay in self.shape:
            variables.extend(lay.trainable_variables())
        gradients = tape.gradient(current_loss, variables)  # Compute gradients
        assert all(g is not None for g in gradients)
        assert len(gradients) % 2 == 0, "No biases still not supported."  # TODO: what if you have no bias? This crashes
        with self.gradients_summary_writer.as_default():
            for i in range(int(len(gradients) / 2)):
                if gradients[2 * i].dtype == tf.complex64 or gradients[2 * i].dtype == tf.complex128:
                    tf.summary.histogram(name="Gradients_w_" + str(i) + "_real",
                                         data=tf.math.real(gradients[2 * i]), step=self.epochs_done)
                    tf.summary.histogram(name="Gradients_w_" + str(i) + "_imag",
                                         data=tf.math.imag(gradients[2 * i]), step=self.epochs_done)
                    tf.summary.histogram(name="Gradients_b_" + str(i) + "_real",
                                         data=tf.math.real(gradients[2 * i + 1]), step=self.epochs_done)
                    tf.summary.histogram(name="Gradients_b_" + str(i) + "_imag",
                                         data=tf.math.imag(gradients[2 * i + 1]), step=self.epochs_done)
                elif gradients[2 * i].dtype == tf.float32 or gradients[2 * i].dtype == tf.float64:
                    tf.summary.histogram(name="Gradients_w_" + str(i),
                                         data=gradients[2 * i], step=self.epochs_done)
                    tf.summary.histogram(name="Gradients_b_" + str(i),
                                         data=gradients[2 * i + 1], step=self.epochs_done)
                else:
                    logger.error("Input_dtype not supported. Should never have gotten here!", exc_info=True)
                    sys.exit(-1)

    def save(self, loss, acc):
        # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
        # TODO: TypeError: can't pickle _thread._local objects
        # https://github.com/tensorflow/tensorflow/issues/33283
        # loss, acc = self.evaluate(x, y)
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

    # ================
    # Managing strings
    # ================

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
            logger.info(string)
        if filename is not None:
            filename = self.root_dir / filename
            try:
                with open(filename, mode) as file:
                    file.write(string + "\n")
            except FileExistsError:  # TODO: Check if this is the actual error
                logging.error("Same file already exists. Aborting to not override results" + str(filename),
                              exc_info=True)
            except FileNotFoundError:
                logging.error("No such file or directory: " + self.root_dir, exc_info=True)
                sys.exit(-1)

    def _get_str_evaluate(self, epoch, epochs, train_x, train_y, test_x=None, test_y=None, batch=None, batches=None, fast_mode=False) -> str:
        train_loss = None
        train_acc = None
        test_loss = None
        test_acc = None
        if not fast_mode:
            train_loss, train_acc, test_loss, test_acc = self.evaluate_train_and_test(train_x, train_y, test_x, test_y)
        return self._get_loss_and_acc_string(epoch, epochs, train_loss, train_acc, test_loss, test_acc, batch, batches, fast_mode)

    @staticmethod
    def _get_loss_and_acc_string(epoch, epochs, train_loss, train_acc, test_loss=None, test_acc=None, batch=None, batches=None, fast_mode=False) -> str:
        ret_str = "Epoch {0}/{1}".format(epoch, epochs)
        if batch is not None and batches is not None:
            ret_str += "; batch {0}/{1}".format(batch, batches)
        if not fast_mode:
            assert train_acc is not None and train_loss is not None, "Train loss or acc was None with fast_mode False"
            ret_str += " - train loss: {0:.4f} - train accuracy: {1:.2f} %".format(train_loss, train_acc * 100)
            if test_loss is not None and test_acc is not None:
                ret_str += "- test loss: {0:.4f} - test accuracy: {1:.2f} %".format(test_loss, test_acc * 100)
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

    def training_param_summary(self):
        summary_str = ""
        summary_str += self.name + "\n"
        if self.is_complex():
            summary_str += "Complex Network"
        else:
            summary_str += "Real Network"
        total_params = 0
        t = PrettyTable(['Layer (type)', 'Output Shape', 'Param #'])
        for lay in self.shape:
            train_params = 0
            for param in lay.trainable_variables():
                train_params += np.prod(param.shape)
            t.add_row([
                lay.__class__.__name__,
                lay.get_output_shape_description(),
                str(train_params)
            ])
            total_params += train_params
        print(summary_str)
        print(t)
        print("Total Trainable params: " + str(total_params))


if __name__ == '__main__':
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 5000
    n = 100
    coef_correls_list = np.linspace(-0.9, 0.9, 2)  # 4 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 2])
    dataset = dp.CorrelatedGaussianCoeffCorrel(m, n, param_list, debug=False)
    # x_fit = transform_to_real(dataset.x)      # To run real case
    x_fit = dataset.x  # To run complex case

    # Define shape
    cdtype = x_fit.dtype
    if cdtype == np.complex64:
        rdtype = np.float32
    else:
        rdtype = np.float64
    input_size = np.shape(x_fit)[1]
    hidden_size = 100
    output_size = np.shape(dataset.y_train)[1]
    # set_trace()
    shape = [layers.Dense(output_size=hidden_size, input_size=input_size, activation='cart_relu',
                          input_dtype=cdtype, dropout=None),
             layers.Dense(output_size=hidden_size, activation='cart_relu'),
             layers.Dense(output_size=hidden_size, activation='cart_relu'),
             layers.Dense(output_size=hidden_size, activation='cart_relu'),
             layers.Dense(output_size=output_size, activation='softmax_real')]
    # set_trace()
    # Train model
    model = CvnnModel("Testing_dropout", shape, tf.keras.losses.categorical_crossentropy,
                      tensorboard=False, verbose=False)
    # set_trace()
    model.fit(x_fit, dataset.y, validation_split=0.0, batch_size=100, epochs=10,
              verbose=True, save_csv_history=True, fast_mode=False, save_txt_fit_summary=False)
    # start = time.time()
    # model.fit(dataset.x, dataset.y, batch_size=100, epochs=30, verbose=False)
    # end = time.time()
    # print(end - start)

    # Analyze data
    # model.plotter.plot_key(key='accuracy', showfig=False, savefig=True)
    # model.plotter.plot_key(key='loss', library='matplotlib', showfig=False, savefig=True)
    # model.get_confusion_matrix(dataset.x_test, dataset.y_test, save_result=True)
    # set_trace()

# How to comment script header
# https://medium.com/@rukavina.andrei/how-to-write-a-python-script-header-51d3cec13731
__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.2.28'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
