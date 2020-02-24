from abc import ABC, abstractmethod
from itertools import count
import tensorflow as tf
import sys
import cvnn
import logging
import cvnn.activation_functions as act
from cvnn.utils import get_func_name
import numpy as np
from tensorflow.keras import layers
from pdb import set_trace

act_dispatcher = {
    'linear': act.linear,
    'cart_sigmoid': act.cart_sigmoid,
    'cart_elu': act.cart_elu,
    'cart_exponential': act.cart_exponential,
    'cart_hard_sigmoid': act.cart_hard_sigmoid,
    'cart_relu': act.cart_relu,
    'cart_selu': act.cart_selu,
    'cart_softplus': act.cart_softplus,
    'cart_softsign': act.cart_softsign,
    'cart_tanh': act.cart_tanh,
    'cart_softmax': act.cart_softmax,
    'cart_softmax_real': act.cart_softmax_real
}

supported_dtypes = (np.complex64, np.float32)   # , np.complex128, np.float64) Gradients return None when complex128
layer_count = count(0)        # Used to count the number of layers


class ComplexLayer(layers.Layer, ABC):

    def __init__(self, input_size, output_size, input_dtype=np.complex64, output_dtype=np.complex64):
        self.logger = logging.getLogger(cvnn.__name__)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_number = next(layer_count)        # Know it's own number
        if output_dtype == np.complex64 and input_dtype == np.float32:
            # TODO: can't it?
            self.logger.error("Layer::__init__: if input dtype is real output cannot be complex")
            sys.exit(-1)
        if input_dtype not in supported_dtypes:
            self.logger.error("Layer::__init__: unsupported input_dtype " + str(input_dtype))
            sys.exit(-1)
        self.input_dtype = input_dtype
        if output_dtype not in supported_dtypes:
            self.logger.error("Layer::__init__: unsupported output_dtype " + str(output_dtype))
            sys.exit(-1)
        self.output_dtype = output_dtype
        super().__init__()

    def get_input_dtype(self):
        return self.input_dtype

    def _apply_activation(self, act_fun, out):
        """
        Applies activation function `act` to variable `out`
        :param out: Tensor to whom the activation function will be applied
        :param act_fun: function to be applied to out. See the list fo possible activation functions on:
            https://complex-valued-neural-networks.readthedocs.io/en/latest/act_fun.html
        :return: Tensor with the applied activation function
        """
        if act_fun is None:     # No activation function declared
            return out
        elif callable(act_fun):
            if act_fun.__module__ == 'activation_functions' or \
                    act_fun.__module__ == 'tensorflow.python.keras.activations':
                return act_fun(out)  # TODO: for the moment is not be possible to give parameters like alpha
            else:
                self.logger.error("Cvnn::_apply_activation Unknown activation function.\n\t "
                                  "Can only use activations declared on activation_functions.py or keras.activations")
                sys.exit(-1)
        elif isinstance(act_fun, str):
            try:
                return act_dispatcher[act_fun](out)
            except KeyError:
                self.logger.warning("WARNING: Cvnn::_apply_function: " + str(act_fun) + " is not callable, ignoring it")
            return out

    @abstractmethod
    def get_description(self):
        pass

    @abstractmethod
    def save_tensorboard_checkpoint(self, summary, step=None):
        pass


class ComplexDense(ComplexLayer):

    def __init__(self, input_size, output_size, activation=None, input_dtype=np.complex64, output_dtype=np.complex64,
                 weight_initializer=tf.keras.initializers.GlorotUniform, bias_initializer=tf.zeros):
        super(ComplexDense, self).__init__(input_size, output_size, input_dtype, output_dtype)
        self.output_dtype = output_dtype
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer  # TODO: Not working yet
        self.w = None
        self.b = None
        self.init_weights()

    def init_weights(self):
        if self.input_dtype == np.complex64 or self.input_dtype == np.complex128:  # Complex layer
            self.w = tf.cast(tf.Variable(tf.complex(self.weight_initializer()(shape=(self.input_size, self.output_size)),
                                                    self.weight_initializer()(shape=(self.input_size, self.output_size))),
                                         name="weights" + str(self.layer_number)),
                             dtype=self.input_dtype)
            self.b = tf.cast(tf.Variable(tf.complex(tf.zeros(self.output_size),
                                                    tf.zeros(self.output_size)),
                                         name="bias" + str(self.layer_number))
                             , dtype=self.input_dtype)
        elif self.input_dtype == np.float32 or self.input_dtype == np.float64:  # Real Layer
            self.w = tf.cast(tf.Variable(self.weight_initializer()(shape=(self.input_size, self.output_size)),
                                         name="weights" + str(self.layer_number)),
                             dtype=self.input_dtype)
            self.b = tf.cast(tf.Variable(tf.zeros(self.output_size),
                                         name="bias" + str(self.layer_number)),
                             dtype=self.input_dtype)
        else:
            # This case should never happen. The abstract constructor should already have checked this
            self.logger.error("Input_dtype not supported.")
            sys.exit(-1)

    def get_description(self):
        fun_name = get_func_name(self.activation)
        out_str = "Dense layer:\n\tinput size = " + str(self.input_size) + "(" + str(self.input_dtype.__name__) + \
                  ") -> output size = " + str(self.output_size) + "(" + str(self.output_dtype.__name__) + \
                  ");\n\tact_fun = " + fun_name + ";\n\tweight init = " \
                  + self.weight_initializer.__name__ + "; bias init = " + self.bias_initializer.__name__ + "\n"
        return out_str

    def call(self, inputs, **kwargs):
        # TODO: treat bias as a weight. It might optimize training (no add operation, only mult)
        with tf.name_scope("ComplexDense_" + str(self.layer_number)) as scope:
            if tf.dtypes.as_dtype(inputs.dtype) is not tf.dtypes.as_dtype(np.dtype(self.input_dtype)):
                self.logger.warning("Dense::apply_layer: Input dtype " + str(inputs.dtype) + " is not as expected ("
                                    + str(tf.dtypes.as_dtype(np.dtype(self.input_dtype))) + "). Trying cast")
            out = tf.add(tf.matmul(tf.cast(inputs, self.input_dtype), self.w), self.b)
            y_out = self._apply_activation(self.activation, out)
            if tf.dtypes.as_dtype(np.dtype(self.output_dtype)) != y_out.dtype:  # Case for real output / real labels
                self.logger.warning("Dense::apply_layer: Automatically casting output")
            return tf.cast(y_out, tf.dtypes.as_dtype(np.dtype(self.output_dtype)))

    def call_v1(self, inputs):
        return self.call(inputs), [self.w, self.b]

    def save_tensorboard_checkpoint(self, summary, step=None):
        with summary.as_default():
            if self.input_dtype == np.complex64 or self.input_dtype == np.complex128:
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_w_real",
                                     data=tf.math.real(self.w), step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_w_imag",
                                     data=tf.math.imag(self.w), step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_b_real",
                                     data=tf.math.real(self.b), step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_b_imag",
                                     data=tf.math.imag(self.b), step=step)
            elif self.input_dtype == np.float32 or self.input_dtype == np.float64:
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_w",
                                     data=self.w, step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_b",
                                     data=self.b, step=step)
            else:
                # This case should never happen. The constructor should already have checked this
                self.logger.error("Input_dtype not supported.")
                sys.exit(-1)


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.18'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
