import os
import sys
import copy
import logging
import numpy as np
import pandas as pd
from itertools import count  # To count the number of times fit is called
import tensorflow as tf
from datetime import datetime
from pdb import set_trace
from time import strftime, perf_counter, gmtime
# My own module!
import cvnn.layers as layers
from cvnn.optimizers import get_optimizer
import cvnn.dataset as dp
import cvnn.data_analysis as da
from cvnn.utils import create_folder
from cvnn import logger
# Typing
from cvnn.optimizers import t_optimizer
from tensorflow.keras.losses import Loss
from typing import Union, Optional, Tuple
from tensorflow import Tensor, data
from numpy import ndarray
from cvnn.layers import t_layers_shape

try:
    from prettytable import PrettyTable
    PRETTY_TABLE = True
except ImportError:
    PRETTY_TABLE = False

try:
    import cPickle as pickle
except ImportError:
    import pickle

t_List = Union[ndarray, list]
t_input_features = Union[t_List, Tensor, data.Dataset]
t_labels = Union[t_List, Tensor]
t_verbose = Union[str, int, bool]

VERBOSITY = {0: "SILENT",  # verbosity 0. NADA DE NADA
             2: "FAST",    # verbosity 2. Muestra al final de cada epoch
             4: "PROBAR",  # Shows the progress bar but without acc or loss
             1: "INFO",    # verbosity 1. Muestra entre cada linea
             3: "DEBUG"
             }


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

    def __init__(self, name: str, shape: t_layers_shape, loss_fun: Loss, optimizer: t_optimizer = 'sgd',
                 verbose: t_verbose = True, tensorboard: bool = True):
        """
        Constructor
        :param name: Name of the model. It will be used to distinguish models
        :param shape: List of cvnn.layers.ComplexLayer objects
        :param loss_fun: tensorflow.python.keras.losses to be used.
        :param optimizer: Optimizer to be used. Keras optimizers are not allowed.
            Can be either cvnn.optimizers.Optimizer or a string listed in opt_dispatcher.
        :param verbose: if True it will print information of np.prod(w_vals.shape)the model just created
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
        layers.ComplexLayer.last_layer_output_dtype = None
        layers.ComplexLayer.last_layer_output_size = None
        self.shape = shape
        self.loss_fun = loss_fun
        self.optimizer = get_optimizer(optimizer)       # This checks input already
        self.optimizer.compile(shape=shape)
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
        self._manage_string(self.summary(), self._get_verbose(verbose), filename=self.name + "_metadata.txt", mode="x")
        self.plotter = da.Plotter(self.root_dir)

    def call(self, x: t_input_features) -> Tensor:        # TODO: Beware, I think tf.Dataset will break here!
        """
        Forward result of the network
        :param x: Data input to be calculated
        :return: Output of the netowrk
        """
        for i in range(len(self.shape)):  # Apply all the layers
            x = self.shape[i].call(x)
        return x

    def _apply_loss(self, y_true: t_labels, y_pred: t_labels) -> Tensor:
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

    def is_complex(self) -> bool:
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
        return CvnnModel(self.name, new_shape, self.loss_fun, optimizer=self.optimizer.__deepcopy__(),
                         verbose=False, tensorboard=self.tensorboard)

    def _get_real_equivalent_multiplier(self, classifier: bool = True, capacity_equivalent: bool = True,
                                        equiv_technique: str = 'ratio'):
        """
        Returns an array (output_multiplier) of size self.shape (number of hidden layers + output layer)
            one must multiply the real valued equivalent layer
        In other words, the real valued equivalent layer 'i' will have:
            neurons_real_valued_layer[i] = output_multiplier[i] * neurons_complex_valued_layer[i]
        :param classifier: Boolean (default = True) weather the model's task is a to classify (True) or
                                                                                            a regression task (False)
        :param capacity_equivalent: An equivalent model can be equivalent in terms of layer neurons or
                        trainable parameters (capacity equivalent according to (https://arxiv.org/abs/1811.12351)
            - True, it creates a capacity-equivalent model in terms of trainable parameters
            - False, it will double all layer size (except the last one if classifier=True)
        :param equiv_technique: Used to define the strategy of the capacity equivalent model.
            This parameter is ignored if capacity_equivalent=False
            - 'ratio': neurons_real_valued_layer[i] = r * neurons_complex_valued_layer[i], 'r' constant for all 'i'
            - 'alternate': Method described in https://arxiv.org/abs/1811.12351 where one alternates between
                    multiplying by 2 or 1. Special case on the middle is treated as a compromise between the two.
        :return: output_multiplier
        """
        if capacity_equivalent:
            if equiv_technique == "alternate":
                output_multiplier = self._get_alternate_capacity_equivalent(classifier)
            elif equiv_technique == "ratio":
                output_multiplier = self._get_ratio_capacity_equivalent(classifier)
            else:
                logger.error("Unknown equiv_technique " + equiv_technique)
                sys.exit(-1)
        else:
            output_multiplier = 2 * np.ones(len(self.shape)).astype(int)
            if classifier:
                output_multiplier[-1] = 1
        return output_multiplier

    def _get_ratio_capacity_equivalent(self, classification: bool = True, bias_adjust: bool = True):
        """
        Generates output_multiplier keeping not only the same capacity but keeping a constant ratio between the
                                                                                                        model's layers
        This helps keeps the 'aspect' or shape of the model my making:
            neurons_real_layer_i = ratio * neurons_complex_layer_i
        :param classification: True (default) if the model is a classification model. False otherwise.
        :param bias_adjust: True (default) if taking into account the bias as a trainable parameter. If not it will
            only match the real valued parameters of the weights
        """
        model_in_c = self.shape[0].input_size
        model_out_c = self.shape[-1].output_size
        x_c = [self.shape[i].output_size for i in range(len(self.shape[:-1]))]
        p_c = np.sum([2 * x.input_size * x.output_size for x in self.shape])  # real valued complex trainable params
        if bias_adjust:
            p_c = p_c + 2 * np.sum(x_c) + 2 * model_out_c
        model_in_r = 2 * model_in_c
        model_out_r = model_out_c if classification else 2 * model_out_c
        # Quadratic equation
        if len(x_c) > 1:
            quadratic_c = -p_c
            quadratic_b = model_in_r * x_c[0] + model_out_r
            if bias_adjust:
                quadratic_b = quadratic_b + np.sum(x_c) + model_out_c
            quadratic_a = np.sum([x_c[i] * x_c[i + 1] for i in range(len(x_c) - 1)])

            ratio = (-quadratic_b + np.sqrt(quadratic_b ** 2 - 4 * quadratic_c * quadratic_a)) / (2 * quadratic_a)
            # The result MUST be positive so I use the '+' solution
            if not 1 <= ratio <= 2:
                logger.error("Ratio {} has a weird value. This function must have a bug.".format(ratio))
        else:
            ratio = 2 * (model_in_c + model_out_c) / (model_in_r + model_out_r)     # TODO: Verify
        return [ratio] * len(x_c) + [1 if classification else 2]

    def _get_alternate_capacity_equivalent(self, classification: bool = True):
        """
        Generates output_multiplier using the alternate method described in https://arxiv.org/abs/1811.12351 which
            doubles or not the layer if it's neighbor was doubled or not (making the opposite).
        The code fills output_multiplier from both senses:
            output_multiplier = [ ... , .... ]
                          --->     <---
        If when both ends meet there's not a coincidence (example: [..., 1, 1, ...]) then
            the code will find a compromise between the two to keep the same real valued trainable parameters.
        """
        output_multiplier = np.zeros(len(self.shape) + 1)
        output_multiplier[0] = 2
        output_multiplier[-1] = 1 if classification else 2
        i: int = 1
        while i <= len(self.shape) - i:
            output_multiplier[i] = 2 if output_multiplier[i - 1] == 1 else 1  # From beginning
            output_multiplier[-1 - i] = 2 if output_multiplier[-i] == 1 else 1  # From the end
            if i == len(self.shape) - i and output_multiplier[i - 1] != output_multiplier[i + 1] or \
                    i + 1 == len(self.shape) - i and output_multiplier[i] == output_multiplier[i + 1]:
                m_inf = self.shape[i - 1].output_size
                m_sup = self.shape[i + 1].output_size
                output_multiplier[i] = 2 * (m_inf + m_sup) / (m_inf + 2 * m_sup)
            i += 1
        return output_multiplier[1:]

    def get_real_equivalent(self, classifier: bool = True, capacity_equivalent: bool = True,
                            equiv_technique: str = 'ratio', name: Optional[str] = None):
        """
        Creates a new model equivalent of current model. If model is already real throws and error.
        :param classifier: True (default) if the model is a classification model. False otherwise.
        :param capacity_equivalent: An equivalent model can be equivalent in terms of layer neurons or
                        trainable parameters (capacity equivalent according to: https://arxiv.org/abs/1811.12351)
            - True, it creates a capacity-equivalent model in terms of trainable parameters
            - False, it will double all layer size (except the last one if classifier=True)
        :param equiv_technique: Used to define the strategy of the capacity equivalent model.
            This parameter is ignored if capacity_equivalent=False
            - 'ratio': neurons_real_valued_layer[i] = r * neurons_complex_valued_layer[i], 'r' constant for all 'i'
            - 'alternate': Method described in https://arxiv.org/abs/1811.12351 where one alternates between
                    multiplying by 2 or 1. Special case on the middle is treated as a compromise between the two.
        :param name: name of the new network to be created.
            If None (Default) it will use same name as current model with "_real_equiv" suffix
        :return: CvnnModel() real equivalent model
        """
        if not self.is_complex():
            logger.error("model {} was already real".format(self.name))
            sys.exit(-1)
        equiv_technique = equiv_technique.lower()
        if equiv_technique not in {"ratio", "alternate"}:
            logger.error("Invalid `equivalent_technique` argument: " + equiv_technique)
            sys.exit(-1)
        # assert len(self.shape) != 0
        real_shape = []
        layers.ComplexLayer.last_layer_output_dtype = None
        layers.ComplexLayer.last_layer_output_size = None
        output_multiplier = self._get_real_equivalent_multiplier(classifier, capacity_equivalent, equiv_technique)
        for i, layer in enumerate(self.shape):
            if isinstance(layer, layers.ComplexLayer):
                if isinstance(layer, layers.Dense):  # TODO: Check if I can do this with kargs or sth
                    real_shape.append(layer.get_real_equivalent(
                        output_multiplier=output_multiplier[i],
                        input_multiplier=output_multiplier[i - 1] if i > 0 else 2))
                else:
                    real_shape.append(layer.get_real_equivalent())
            else:
                sys.exit("Layer " + str(layer) + " unknown")
        if name is None:
            name = self.name + "_real_equiv"
        # set_trace()
        return CvnnModel(name=name, shape=real_shape, loss_fun=self.loss_fun, optimizer=self.optimizer.__deepcopy__(),
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

    # Add '@tf.function' to accelerate the code by a lot!
    @tf.function
    def _train_step(self, x_train_batch: Tensor, y_train_batch:  Tensor):
        """
        Performs one step of the training
        :param x_train_batch: input
        :param y_train_batch: labels
        :return: None
        """
        with tf.GradientTape() as tape:
            with tf.name_scope("Forward_Phase"):
                x_called = self.call(x_train_batch)  # Forward mode computation
            # Loss function computation
            with tf.name_scope("Loss"):
                current_loss = self._apply_loss(y_train_batch, x_called)  # Compute loss

        # Calculating gradient
        with tf.name_scope("Gradient"):
            variables = []
            for lay in self.shape:
                variables.extend(lay.trainable_variables())  # TODO: Debug this for all layers.
            gradients = tape.gradient(current_loss, variables)  # Compute gradients
            assert all(g is not None for g in gradients)

        # Backpropagation
        with tf.name_scope("Optimizer"):
            self.optimizer.optimize(variables=variables, gradients=gradients)

    @run_once
    def _end_graph_tensorflow(self):
        with self.graph_writer.as_default():
            tf.summary.trace_export(name="graph", step=0, profiler_outdir=self.graph_writer_logdir)

    def fit(self, x: t_input_features, y: Optional[t_labels] = None,
            validation_split: float = 0.0, validation_data: Optional[Union[Tuple[t_List], data.Dataset]] = None,
            epochs: int = 10, batch_size: int = 32,
            verbose: t_verbose = True, display_freq: int = 1,
            save_model_checkpoints: bool = False, save_csv_history: bool = True, shuffle: bool = True):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        :param x: Input data. It could be:
            - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            - A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            - A tf.data dataset. Should return a tuple (inputs, targets). Preferred data type (less overhead).
        :param y: Labels/Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s).
            If f x is a dataset then y will be ignored (default None)
        :param validation_split: Float between 0 and 1.
            Percentage of the input data to be used as test set (the rest will be use as train set)
            Default: 0.0 (No validation set).
            This input is ignored if validation_data is given.
        :param validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. This parameter takes precedence over validation_split.
            It can be:
                - tuple (x_val, y_val) of Numpy arrays or tensors. Preferred data type (less overhead).
                - A tf.data dataset.
        :param epochs: (uint) Number of epochs to do.
        :param batch_size: (uint) Batch size of the data. Default 32 (because keras use 32 so... why not?)
        :param verbose: Verbosity Mode
            It can be:
                - Bool: False defaults to 0 and True to 1.
                - Int
                - String: Matching the modes string
            Verbosity Modes:
                - "SILENT" or 0:  No prints of any kind
                - "FAST" or 2:    Does not show the progress bar of each epoch.
                    Verbosity modes "FAST" and "SILENT" saves the csv file (if save_csv_history) less often.
                    Making it faster riskier of data loss
                - "PROBAR" or 4:  Shows progress bar but does not show accuracy or loss (helps on speed)
                - "INFO" or 1:    Shows a progress bar with current accuracy and loss
                - "DEBUG" or 3:   Shows start and end messages and also the progress bar with current accuracy and loss
            Verbosity modes 0, 1 and 2 are coincident with tensorflow's fit verbose parameter:
                https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        :param display_freq: Integer (Default 1)
            Frequency on terms of epochs before saving information and running a checkpoint.
        :param save_model_checkpoints: (Boolean)
                    Save the model to be able to load and continue training later TODO: Not yet working
        :param save_csv_history: (Boolean) Save information of the train and test loss and accuracy on csv files.
        :param shuffle: (Boolean) Whether to shuffle the training data before each epoch. Default: True
        :return: None
        """
        # Check input
        verbose = self._verify_fit_input(save_model_checkpoints, epochs, batch_size, display_freq, verbose)
        # Prepare dataset
        train_dataset, (x_test, y_test) = self._process_dataset(x, y, validation_split, validation_data, batch_size)
        # train_dataset = train_dataset.batch(batch_size=batch_size)  # TODO: Check if batch_size = 1
        # Print start condition
        start_status = ''
        if x_test is not None and y_test is not None:
            start_status = self._get_str_evaluate(self.epochs_done, epochs, x_test, y_test)
        self._manage_string("Starting training...\n" +
                            "Epochs = " + str(epochs) + "\nBatch Size = " + str(batch_size) + "\n" +
                            start_status, verbose)
        # -----------------------------------------------------
        # input processing ended
        # num_tr_iter = int(x_train.shape[0] / batch_size)  # Number of training iterations in each epoch
        epochs_before_fit = self.epochs_done
        start_time = perf_counter()
        total_iteration = None
        for epoch in range(epochs):
            iteration = 0
            if verbose in ("FAST", "INFO", "DEBUG", "PROBAR"):
                tf.print("Epoch {0}/{1}".format(self.epochs_done + 1, epochs))
                progbar = tf.keras.utils.Progbar(total_iteration,
                                                 stateful_metrics=[('loss', 0), ('accuracy', 0),
                                                                   ('val_loss', 0), ('val_accuracy', 0)])
            # Randomly shuffle the training data at the beginning of each epoch
            if shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=50000)
            for x_batch, y_batch in train_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache():
                if verbose in ("INFO", "DEBUG", "PROBAR"):
                    values = None
                    if verbose != "PROBAR":
                        train_loss, train_acc = self.evaluate_train_and_test(x_batch, y_batch)[:2]
                        values = [('loss', train_loss), ('accuracy', train_acc)]
                    progbar.update(iteration, values=values)
                iteration += 1
                # Run optimization op (backpropagation)
                self._start_graph_tensorflow()
                self._train_step(x_batch, y_batch)
                self._end_graph_tensorflow()
            if total_iteration is None:
                total_iteration = iteration
            if verbose in ("FAST", "INFO", "DEBUG", "PROBAR"):
                values = None
                if verbose != "PROBAR":
                    train_loss, train_acc, test_loss, test_acc = self.evaluate_train_and_test(x_batch, y_batch,
                                                                                              x_test,  y_test)
                    values = [('loss', train_loss), ('accuracy', train_acc)]
                    if test_loss is not None:
                        values += [('val_loss', test_loss), ('val_accuracy', test_acc)]
                progbar.update(iteration,
                               values=values,
                               finalize=True)
            # Save checkpoint if needed
            if (epochs_before_fit + epoch) % display_freq == 0:
                self._run_checkpoint(x_batch, y_batch, x_test, y_test,  # Shall I use batch to be more efficient?
                                     step=(epochs_before_fit + epoch) * total_iteration + total_iteration,
                                     num_tr_iter=total_iteration, total_epochs=epochs_before_fit + epochs,
                                     verbose="SILENT", save_model_checkpoints=save_model_checkpoints,
                                     save_csv_checkpoints=save_csv_history,
                                     fast_mode=True if verbose in ("SILENT", "FAST") else False)
            self.epochs_done += 1

        # After epochs
        end_time = perf_counter()
        x_train = x_batch
        y_train = y_batch
        self._run_checkpoint(x_train, y_train, x_test, y_test,
                             step=epochs * total_iteration + total_iteration, num_tr_iter=total_iteration,
                             total_epochs=epochs_before_fit + epochs,
                             fast_mode=False)       # I use false to save the csv file this time
        end_status = self._get_str_evaluate(epochs, epochs, x_train, y_train, x_test, y_test)
        self._manage_string("Train finished...\n" +
                            end_status +
                            "\nTraining time: {}s".format(strftime("%H:%M:%S", gmtime(end_time - start_time))),
                            verbose)
        self.plotter.reload_data()

    def _verify_fit_input(self, save_model_checkpoints: bool, epochs: int, batch_size: int, display_freq: int,
                          verbose: t_verbose) -> str:
        assert not save_model_checkpoints  # TODO: Not working for the moment, sorry!
        if not (isinstance(epochs, int) and epochs > 0):
            logger.error("Epochs must be unsigned integer", exc_info=True)
            sys.exit(-1)
        if not (isinstance(batch_size, int) and batch_size > 0):
            logger.error("Batch size must be unsigned integer", exc_info=True)
            sys.exit(-1)
        if isinstance(display_freq, int):
            assert display_freq > 0, "display_freq must be positive"
        else:
            logger.error("display_freq must be a unsigned integer. Got" + str(display_freq))
            sys.exit(-1)
        return self._get_verbose(verbose)

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

    @staticmethod
    def _get_verbose(verbose: t_verbose) -> str:
        if isinstance(verbose, bool):
            return "INFO" if verbose else "SILENT"
        elif isinstance(verbose, int):
            return VERBOSITY[verbose]
        elif isinstance(verbose, str):
            if verbose.upper() in VERBOSITY.values():
                return verbose.upper()
            else:
                supported_strings = "\n\t"
                for s in VERBOSITY.values():
                    supported_strings += s + "\n\t"
                logger.error("verbose string unknown \"" + verbose + "\". Supported: " + supported_strings)
                sys.exit(-1)
        else:
            logger.error("verbose datatype (" + str(type(verbose)) + ") not supported")

    @staticmethod
    def _process_dataset(x: t_input_features, y: Optional[t_labels],
                         validation_split: float = 0.0,
                         validation_data: Optional[Union[Tuple[t_List], data.Dataset]] = None,
                         batch_size: int = 32) -> Tuple[data.Dataset, Tuple]:
        """
        Process dataset and returns in the preferred format.

        :param x: Input data. It could be:
            - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            - A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            - A tf.data dataset. Should return a tuple (inputs, targets). Preferred data type (less overhead).
        :param y: Labels/Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s).
            If f x is a dataset then y will be ignored (default None)
        :param validation_split: Float between 0 and 1.
            Percentage of the input data to be used as test set (the rest will be use as train set)
            Default: 0.0 (No validation set).
            This input is ignored if validation_data is given.
        :param validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. This parameter takes precedence over validation_split.
            It can be:
                - tuple (x_val, y_val) of Numpy arrays or tensors. Preferred data type (less overhead).
                - A tf.data dataset.
        :param batch_size: (uint) Batch size of the data. Default 32 (because keras use 32 so... why not?)
        :returns: (train, test) with train a tf.data.Dataset and test a tuple (x_val, y_val)
        """
        test_dataset = None
        if isinstance(x, (list, tuple, np.ndarray)):
            if validation_data is None:
                assert 0 <= validation_split < 1, "Ratio should be between [0, 1)"
                dataset_length = np.shape(x)[0]
                x_train = x[int(dataset_length * validation_split):]
                y_train = y[int(dataset_length * validation_split):]
                x_test = x[:int(dataset_length * validation_split)]
                y_test = y[:int(dataset_length * validation_split)]
                if len(x_test) != 0:
                    validation_data = (np.array(x_test), np.array(y_test))
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=batch_size)
            else:
                if validation_split != 0:
                    logger.warning("validation_split was given but will be ignored because "
                                   "validation_data was not None")
                train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=batch_size)
        elif isinstance(x, tf.data.Dataset):
            if y is not None:
                logger.warning("y is ignored because x was a Dataset (and should contain the labels), "
                               "however, y was not None")
            train_dataset = x
        else:
            logger.error("dataset type ({}) not supported".format(type(x)))
            sys.exit(-1)
        if validation_data is not None:
            if isinstance(validation_data, (list, tuple, np.ndarray)):
                assert len(validation_data) == 2, \
                    "validation_data must have size 2. Size was {}".format(len(validation_data))
                test_dataset = np.array(validation_data[0]), np.array(validation_data[1])
            elif isinstance(validation_data, tf.data.Dataset):
                x_test = []
                y_test = []
                for element in validation_data.unbatch():
                    x_test.append(element[0].numpy())
                    y_test.append(element[1].numpy())
                test_dataset = [np.array(x_test), np.array(y_test)]
            else:
                logger.error("validation_data type ({}) not supported ".format(type(validation_data)))
                sys.exit(-1)
        assert isinstance(test_dataset, (list, tuple, np.ndarray)) or test_dataset is None
        if test_dataset is None:
            test_dataset = (None, None)
        return train_dataset, test_dataset

    @staticmethod
    def _dataset_to_array(x: t_input_features, y: Optional[t_labels] = None) -> Tuple[ndarray, ndarray]:
        if isinstance(y, (list, tuple, np.ndarray, tf.Tensor)):
            return np.array(x), np.array(y)
        elif isinstance(x, tf.data.Dataset):
            x_test = []
            y_test = []
            for element in x.unbatch():
                x_test.append(element[0].numpy())
                y_test.append(element[1].numpy())
            return np.array(x_test), np.array(y_test)
        else:
            logger.error("dataset type ({}) not supported ".format(type(x)))
            sys.exit(-1)

    @classmethod  # https://stackoverflow.com/a/2709848/5931672
    def loader(cls, f):
        return pickle.load(f)  # TODO: not yet tested

    # ==========================
    # Predict models and results
    # ==========================

    def predict(self, x: t_input_features) -> Tensor:
        """
        Predicts the value of the class.
        ATTENTION: Use this only for classification tasks. For regression use "call" method.
        :param x: Input
        :return: Prediction of the class that x belongs to.
        """
        y_out = self.call(x)
        return tf.math.argmax(y_out, 1)

    def evaluate_loss(self, x: t_input_features, y: Optional[t_labels] = None) -> ndarray:
        """
        Computes the output of x and computes the loss using y
        :param x: Input of the netwotk
        :param y: Labels
        :return: loss value
        """
        x, y = self._dataset_to_array(x, y)
        return self._apply_loss(y, self.call(x)).numpy()

    def evaluate_accuracy(self, x: t_input_features, y: Optional[t_labels] = None) -> ndarray:
        """
        Computes the output of x and returns the accuracy using y as labels
        :param x: Input of the netwotk
        :param y: Labels
        :return: accuracy
        """
        x, y = self._dataset_to_array(x, y)
        y_pred = self.predict(x)
        if y_pred.shape == y.shape:
            y_labels = y
        else:
            y_labels = tf.math.argmax(y, 1)
        return tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(y_pred, y_labels), tf.float64)).numpy()

    def evaluate(self, x, y=None):
        """
        Computes both the loss and accuracy using "evaluate_loss" and "evaluate_accuracy"
        :param x: Input of the network
        :param y: Labels
        :return: tuple (loss, accuracy)
        """
        x, y = self._dataset_to_array(x, y)
        return self.evaluate_loss(x, y), self.evaluate_accuracy(x, y)

    def evaluate_train_and_test(self, train_x, train_y, test_x=None, test_y=None):
        train_loss, train_acc = self.evaluate(train_x, train_y)
        test_loss, test_acc = None, None
        if test_x is not None and test_y is not None:
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
        # set_trace()
        return da.confusion_matrix(self.call(x), y, filename=filename)

    # ===========
    # Checkpoints
    # ===========

    def _run_checkpoint(self, x_train, y_train, x_test=None, y_test=None,
                        step=0, num_tr_iter=0, total_epochs=0, verbose="SILENT",
                        save_model_checkpoints=False, save_csv_checkpoints=True, fast_mode=False):
        """
        Saves whatever needs to be saved (tensorboard, csv of train and test acc and loss, model weigths, etc.
        :param x_train: Train data
        :param y_train: Tran labels
        :param x_test: Test data (optional)
        :param y_test: Test labels (optional)
        :param step: step of the training.
        :param num_tr_iter: Total number of iterations per epoch.
        :param total_epochs: Total epochs to be done on the training.
        :param fast_mode: if True it will save the loss and accuracy as a csv on each file.
                This will prevent the loss of data but will make the training longer.
                Default: False.
        :param verbose: Print the results on console to visualize the training step.
        :param save_model_checkpoints: Save the model to be able to load and continue training later (Not yet working)
        :param save_csv_checkpoints: Save information of the train and test loss and accuracy on csv files.
        :return: None
        """
        # First I check if at least one is needed. If not better don't compute the information.
        if save_csv_checkpoints or save_model_checkpoints or verbose:
            train_loss, train_acc, test_loss, test_acc = self.evaluate_train_and_test(x_train, y_train, x_test, y_test)
        if self.tensorboard:  # Save tensorboard data
            self._tensorboard_checkpoint(x_train, y_train, x_test, y_test)
        if save_csv_checkpoints:
            # With fast mode False I save the checkpoint in a csv.
            # It will take longer to run because I create a file each time
            # but if I don't do it and something happens I will loose all the information
            self._save_current_loss_and_acc(self.name + '_results_fit',
                                            train_loss, train_acc, test_loss, test_acc, step, fast_mode=fast_mode)
        if save_model_checkpoints:  # Save model weights
            self.save(test_loss, test_acc)
        if verbose != "SILENT":  # I first check if it makes sense to get the str
            epoch_str = self._get_loss_and_acc_string(epoch=self.epochs_done, epochs=total_epochs,
                                                      train_loss=train_loss, train_acc=train_acc,
                                                      test_loss=test_loss, test_acc=test_acc,
                                                      batch=step % num_tr_iter, batches=num_tr_iter)
            self._manage_string(epoch_str, verbose, None)

    def _save_current_loss_and_acc(self, filename, train_loss, train_acc, test_loss, test_acc, step, fast_mode=False):
        """
        Adds the accuracy and loss result of current step to the pandas backup history and saves it (optional) into csv
        :param filename: File name where the data should be saved.
        :param train_loss:
        :param train_acc:
        :param test_loss:
        :param test_acc:
        :param step:
        :param fast_mode: Bool (Default False). If True, the file will not be saved.
        """
        a_series = pd.Series([step, self.epochs_done, train_loss, train_acc, test_loss, test_acc],
                             index=self.run_pandas.columns)
        self.run_pandas = self.run_pandas.append(a_series, ignore_index=True)
        if not fast_mode:
            if not filename.endswith('.csv'):
                filename += '.csv'
            filename = self.root_dir / filename
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

    def _save_tensorboard_gradients(self, x_train, y_train) -> None:
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

    def save(self, loss, acc) -> None:
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

    def _manage_string(self, string, verbose="SILENT", filename=None, mode="a") -> None:
        """
        Prints a string to console and/or saves it to a file
        :param string: String to be printed/saved
        :param verbose: String (Default 'SILENT'). If "DEBUG" it will print the message into the logger as info.
        :param filename: Filename where to save the string. If None it will not save it (default: None)
        :param mode: Mode to open the filename
        :return: None
        """
        if verbose in ("DEBUG",):
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

    def _get_str_evaluate(self, epoch, epochs, x, y, x_test=None, y_test=None, batch=None, batches=None) -> str:
        train_loss, train_acc, test_loss, test_acc = self.evaluate_train_and_test(x, y, x_test, y_test)
        return self._get_loss_and_acc_string(epoch, epochs, train_loss, train_acc, test_loss, test_acc, batch, batches)

    @staticmethod
    def _get_loss_and_acc_string(epoch: int, epochs: int, train_loss: float, train_acc: float,
                                 test_loss: float = None, test_acc: float = None, batch: int = None,
                                 batches: int = None) -> str:
        ret_str = "Epoch {0}/{1}".format(epoch, epochs)
        if batch is not None and batches is not None:
            ret_str += "; batch {0}/{1}".format(batch, batches)
        assert train_acc is not None and train_loss is not None, "Train loss or acc was None with fast_mode False"
        ret_str += " - train loss: {0:.4f} - train accuracy: {1:.2f} %".format(train_loss, train_acc * 100)
        if test_loss is not None and test_acc is not None:
            ret_str += "- test loss: {0:.4f} - test accuracy: {1:.2f} %".format(test_loss, test_acc * 100)
        return ret_str

    def summary(self) -> str:
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
        summary_str += "Loss function: " + self.loss_fun.__name__ + "\n"
        summary_str += self.optimizer.summary()
        for lay in self.shape:
            summary_str += lay.get_description()
        return summary_str

    def training_param_summary(self) -> None:
        """
        Prints a table analog to tf.keras.Model.summary()
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary
        """
        if not PRETTY_TABLE:
            logger.warning("Function CvnnModel.training_param_summary() was called "
                           "but PrettyTable is not installed so it will be omitted")
            return None
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
    model.fit(x_fit, dataset.y, validation_split=0.0, batch_size=100, epochs=3,
              verbose=True, save_csv_history=True)
    model.call(x_fit)
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
__version__ = '0.2.56'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
