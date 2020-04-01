from abc import ABC, abstractmethod
from itertools import count
import tensorflow as tf
import sys
import cvnn
import logging
from cvnn.activation_functions import apply_activation
from cvnn.utils import get_func_name
import numpy as np
from tensorflow.keras import layers
from pdb import set_trace

# Initializers:
# https://www.tensorflow.org/api_docs/python/tf/keras/initializers
# https://keras.io/initializers/

supported_dtypes = (np.complex64, np.float32)   # , np.complex128, np.float64) Gradients return None when complex128
layer_count = count(0)                          # Used to count the number of layers


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

    @abstractmethod
    def get_description(self):
        pass

    @abstractmethod
    def save_tensorboard_checkpoint(self, summary, step=None):
        pass


class ComplexDense(ComplexLayer):
    """
    Fully connected complex-valued layer
    Implements the operation:
        activation(dot(input, weights) + bias)
    - where data types can be either complex or real.
    - activation is the element-wise activation function passed as the activation argument,
    - weights is a matrix created by the layer
    - bias is a bias vector created by the layer
    """

    def __init__(self, input_size, output_size, activation=None, input_dtype=np.complex64, output_dtype=np.complex64,
                 weight_initializer=tf.keras.initializers.GlorotUniform, bias_initializer=tf.keras.initializers.Zeros):
        """
        Initializer of the Dense layer
        :param input_size: Input size of the layer  # TODO: make it compulsory only for the first layer.
        :param output_size: Output size of the layer
        :param activation: Activation function to be used.
            Can be either the function from cvnn.activation or tensorflow.python.keras.activations
            or a string as listed in act_dispatcher
        :param input_dtype: data type of the input. Default: np.complex64
            Supported data types:
                - np.complex64
                - np.float32
        :param output_dtype: data type of the output function.
            Default: np.complex64   # TODO: Shall I make it not necessary? Let the activation function decide it.
        :param weight_initializer: Initializer for the weights.
            Default: tensorflow.keras.initializers.GlorotUniform
        :param bias_initializer: Initializer fot the bias.
            Default: tensorflow.keras.initializers.Zeros
        """
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
            self.b = tf.cast(tf.Variable(tf.complex(self.bias_initializer()(self.output_size),
                                                    self.bias_initializer()(self.output_size)),
                                         name="bias" + str(self.layer_number))
                             , dtype=self.input_dtype)
        elif self.input_dtype == np.float32 or self.input_dtype == np.float64:  # Real Layer
            self.w = tf.cast(tf.Variable(self.weight_initializer()(shape=(self.input_size, self.output_size)),
                                         name="weights" + str(self.layer_number)),
                             dtype=self.input_dtype)
            self.b = tf.cast(tf.Variable(self.bias_initializer()(self.output_size),
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
            y_out = apply_activation(self.activation, out)
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
__version__ = '0.0.20'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
