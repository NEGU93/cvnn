from abc import ABC, abstractmethod
from itertools import count
import tensorflow as tf
from collections.abc import Iterable
import sys
from cvnn.activation_functions import apply_activation
from cvnn.utils import get_func_name
import numpy as np
from cvnn import logger
from time import time
from pdb import set_trace
import cvnn.initializers as initializers
# Typing
from tensorflow import dtypes
from numpy import dtype, ndarray
from typing import Union, Callable, Optional, List, Set


SUPPORTED_DTYPES = (np.complex64, np.float32)  # , np.complex128, np.float64) Gradients return None when complex128
layer_count = count(0)  # Used to count the number of layers

t_input_shape = Union[int, tuple, list]
t_Callable_shape = Union[t_input_shape, Callable]   # Either a input_shape or a function that sets self.output
t_Dtype = Union[dtypes.DType, dtype]


class ComplexLayer(ABC):
    # Being ComplexLayer an abstract class, then this can be called using:
    #   self.__class__.__bases__.<variable>
    # As all child's will have this class as base, mro gives a full list so won't work.
    last_layer_output_dtype = None      # TODO: Make it work both with np and tf dtypes
    last_layer_output_size = None

    def __init__(self, output_size: t_Callable_shape, input_size=Optional[t_input_shape], input_dtype=t_Dtype, **args):
        """
        Base constructor for a complex layer. The first layer will need a input_dtype and input_size.
        For the other classes is optional,
            if input_size or input_dtype does not match last layer it will throw a warning
        :param output_size: Output size of the layer.
            If the output size depends on the input_size, a function must be passed as output_size.
        :param input_size: Input size of the layer
        :param input_dtype: data type of the input
        """
        if output_size is None:
            logger.error("Output size = None not supported")
            sys.exit(-1)

        if input_dtype is None and self.__class__.__bases__[0].last_layer_output_dtype is None:
            # None input dtype given but it's the first layer declared
            logger.error("First layer must be given an input dtype", exc_info=True)
            sys.exit(-1)
        elif input_dtype is None and self.__class__.__bases__[0].last_layer_output_dtype is not None:
            # Use automatic mode
            self.input_dtype = self.__class__.__bases__[0].last_layer_output_dtype
        elif input_dtype is not None:
            if input_dtype not in SUPPORTED_DTYPES:
                logger.error("Layer::__init__: unsupported input_dtype " + str(input_dtype), exc_info=True)
                sys.exit(-1)
            if self.__class__.__bases__[0].last_layer_output_dtype is not None:
                if self.__class__.__bases__[0].last_layer_output_dtype != input_dtype:
                    logger.warning("Input dtype " + str(input_dtype) +
                                        " is not equal to last layer's input dtype " +
                                        str(self.__class__.__bases__[0].last_layer_output_dtype))
            self.input_dtype = input_dtype

        # This will be normally the case.
        # Each layer must change this value if needed.
        self.__class__.__bases__[0].last_layer_output_dtype = self.input_dtype

        # Input Size
        if input_size is None:
            if self.__class__.__bases__[0].last_layer_output_size is None:
                # None input size given but it's the first layer declared
                logger.error("First layer must be given an input size")
                sys.exit(-1)
            else:  # self.__class__.__bases__[0].last_layer_output_dtype is not None:
                self.input_size = self.__class__.__bases__[0].last_layer_output_size
        elif input_size is not None:
            if self.__class__.__bases__[0].last_layer_output_size is not None:
                if input_size != self.__class__.__bases__[0].last_layer_output_size:
                    logger.warning("Input size " + str(input_size) + " is not equal to last layer's output size " +
                                        str(self.__class__.__bases__[0].last_layer_output_size))
            self.input_size = input_size

        if callable(output_size):
            output_size()
            assert self.output_size is not None, "Error: output_size function must set self.output_size"
        else:
            self.output_size = output_size
        for x in self.__class__.mro():
            if x == ComplexLayer:
                x.last_layer_output_size = self.output_size
        # self.__class__.__bases__[0].last_layer_output_size = self.output_size
        self.layer_number = next(layer_count)  # Know it's own number
        self.__class__.__call__ = self.call     # Make my object callable

    @abstractmethod
    def __deepcopy__(self, memodict=None):
        pass

    def get_input_dtype(self):
        return self.input_dtype

    @abstractmethod
    def get_real_equivalent(self):
        """
        :return: Gets a real-valued COPY of the Complex Layer.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        :return: a string containing all the information of the layer
        """
        pass

    def _save_tensorboard_output(self, x, summary, step):
        x = self.call(x)
        with summary.as_default():
            if x.dtype == tf.complex64 or x.dtype == tf.complex128:
                tf.summary.histogram(name="Activation_value_" + str(self.layer_number) + "_real",
                                     data=tf.math.real(x), step=step)
                tf.summary.histogram(name="Activation_value_" + str(self.layer_number) + "_imag",
                                     data=tf.math.imag(x), step=step)
            elif x.dtype == tf.float32 or x.dtype == tf.float64:
                tf.summary.histogram(name="Activation_value_" + str(self.layer_number),
                                     data=x, step=step)
            else:
                logger.error("Input_dtype not supported. Should never have gotten here!", exc_info=True)
                sys.exit(-1)
        return x

    def save_tensorboard_checkpoint(self, x, weight_summary, activation_summary, step=None):
        self._save_tensorboard_weight(weight_summary, step)
        return self._save_tensorboard_output(x, activation_summary, step)

    @abstractmethod
    def _save_tensorboard_weight(self, weight_summary, step):
        pass

    @abstractmethod
    def trainable_variables(self):
        pass

    @abstractmethod
    def call(self, inputs):
        pass

    def get_output_shape_description(self) -> str:
        # output_string = ""
        if isinstance(self.output_size, Iterable):
            output_string = "(None, " + ", ".join([str(x) for x in self.output_size]) + ")"
        else:
            output_string = "(None, " + str(self.output_size) + ")"
        return output_string


class Flatten(ComplexLayer):

    def __init__(self, input_size=None, input_dtype=None):
        # Win x2: giving None as input_size will also make sure Flatten is not the first layer
        super().__init__(input_size=input_size, output_size=self._get_output_size, input_dtype=input_dtype)

    def __deepcopy__(self, memodict=None):
        return Flatten()

    def _get_output_size(self):
        self.output_size = np.prod(self.input_size)

    def get_real_equivalent(self):
        return self.__deepcopy__()

    def get_description(self) -> str:
        return "Complex Flatten"

    def _save_tensorboard_weight(self, weight_summary, step):
        return None

    def call(self, inputs):
        return tf.reshape(inputs, (inputs.shape[0], self.output_size))

    def trainable_variables(self):
        return []


class Dense(ComplexLayer):
    """
    Fully connected complex-valued layer
    Implements the operation:
        activation(dot(input, weights) + bias)
    - where data types can be either complex or real.
    - activation is the element-wise activation function passed as the activation argument,
    - weights is a matrix created by the layer
    - bias is a bias vector created by the layer
    """

    def __init__(self, output_size, input_size=None, activation=None, input_dtype=None,
                 weight_initializer=None, bias_initializer=None,
                 dropout=None):
        """
        Initializer of the Dense layer
        :param output_size: Output size of the layer
        :param input_size: Input size of the layer
        :param activation: Activation function to be used.
            Can be either the function from cvnn.activation or tensorflow.python.keras.activations
            or a string as listed in act_dispatcher
        :param input_dtype: data type of the input. Default: np.complex64
            Supported data types:
                - np.complex64
                - np.float32
        :param weight_initializer: Initializer for the weights.
            Default: tensorflow.keras.initializers.GlorotUniform
        :param bias_initializer: Initializer fot the bias.
            Default: tensorflow.keras.initializers.Zeros
        :param dropout: Either None (default) and no dropout will be applied or a scalar
            that will be the probability that each element is dropped.
            Example: setting rate=0.1 would drop 10% of input elements.
        """
        super(Dense, self).__init__(output_size=output_size, input_size=input_size, input_dtype=input_dtype)
        if activation is None:
            activation = 'linear'
        self.activation = activation
        # Test if the activation function changes datatype or not...
        self.__class__.__bases__[0].last_layer_output_dtype = \
            apply_activation(self.activation,
                             tf.cast(tf.complex([[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]), self.input_dtype)
                             ).numpy().dtype
        self.dropout = dropout      # TODO: I don't find the verification that it is between 0 and 1. I think I omitted
        if weight_initializer is None:
            weight_initializer = initializers.GlorotUniform()
        self.weight_initializer = weight_initializer
        if bias_initializer is None:
            bias_initializer = initializers.Zeros()
        self.bias_initializer = bias_initializer
        self.w = None
        self.b = None
        self._init_weights()

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return Dense(output_size=self.output_size, input_size=self.input_size,
                     activation=self.activation,
                     input_dtype=self.input_dtype,
                     weight_initializer=self.weight_initializer,
                     bias_initializer=self.bias_initializer, dropout=self.dropout
                     )

    def get_real_equivalent(self, output_multiplier=2, input_multiplier = 2):
        """
        :param output_multiplier: Multiplier of output and input size (normally by 2)
        :return: real-valued copy of self
        """
        return Dense(output_size=int(round(self.output_size * output_multiplier)),
                     input_size=int(round(self.input_size * input_multiplier)),
                     activation=self.activation, input_dtype=np.float32,
                     weight_initializer=self.weight_initializer,
                     bias_initializer=self.bias_initializer, dropout=self.dropout
                     )

    def _init_weights(self):
        self.w = tf.Variable(self.weight_initializer(shape=(self.input_size, self.output_size), dtype=self.input_dtype),
                             name="weights" + str(self.layer_number))
        self.b = tf.Variable(self.bias_initializer(shape=self.output_size, dtype=self.input_dtype),
                             name="bias" + str(self.layer_number))

    def get_description(self):
        fun_name = get_func_name(self.activation)
        out_str = "Dense layer:\n\tinput size = " + str(self.input_size) + "(" + str(self.input_dtype) + \
                  ") -> output size = " + str(self.output_size) + \
                  ";\n\tact_fun = " + fun_name + ";\n\tweight init = " \
                  "\n\tDropout: " + str(self.dropout) + "\n"
        # + self.weight_initializer.__name__ + "; bias init = " + self.bias_initializer.__name__ + \
        return out_str

    def call(self, inputs):
        """
        Applies the layer to an input
        :param inputs: input
        :param kwargs:
        :return: result of applying the layer to the inputs
        """
        # TODO: treat bias as a weight. It might optimize training (no add operation, only mult)
        with tf.name_scope("ComplexDense_" + str(self.layer_number)) as scope:
            if tf.dtypes.as_dtype(inputs.dtype) is not tf.dtypes.as_dtype(np.dtype(self.input_dtype)):
                logger.warning("Dense::apply_layer: Input dtype " + str(inputs.dtype) + " is not as expected ("
                                    + str(tf.dtypes.as_dtype(np.dtype(self.input_dtype))) +
                                    "). Casting input but you most likely have a bug")
            out = tf.add(tf.matmul(tf.cast(inputs, self.input_dtype), self.w), self.b)
            y_out = apply_activation(self.activation, out)

            if self.dropout:
                # $ tf.nn.dropout(tf.complex(x,x), rate=0.5)
                # *** ValueError: x has to be a floating point tensor since it's going to be scaled.
                # Got a <dtype: 'complex64'> tensor instead.
                drop_filter = tf.nn.dropout(tf.ones(y_out.shape), rate=self.dropout)
                y_out_real = tf.multiply(drop_filter, tf.math.real(y_out))
                y_out_imag = tf.multiply(drop_filter, tf.math.imag(y_out))
                y_out = tf.cast(tf.complex(y_out_real, y_out_imag), dtype=y_out.dtype)
            return y_out

    def _save_tensorboard_weight(self, summary, step):
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
                logger.error("Input_dtype not supported.", exc_info=True)
                sys.exit(-1)

    def trainable_variables(self):
        return [self.w, self.b]


class Dropout(ComplexLayer):

    def __init__(self, rate, noise_shape=None, seed=None):
        """
        :param rate: A scalar Tensor with the same type as x.
            The probability that each element is dropped.
            For example, setting rate=0.1 would drop 10% of input elements.
        :param noise_shape: A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
        :param seed:  A Python integer. Used to create random seeds. See tf.random.set_seed for behavior.
        """
        # tf.random.set_seed(seed)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        # Win x2: giving None as input_size will also make sure Dropout is not the first layer
        super().__init__(input_size=None, output_size=self.dummy, input_dtype=None)

    def dummy(self):
        self.output_size = self.input_size

    def call(self, inputs):
        drop_filter = tf.nn.dropout(tf.ones(inputs.shape), rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)
        y_out_real = tf.multiply(drop_filter, tf.math.real(inputs))
        y_out_imag = tf.multiply(drop_filter, tf.math.imag(inputs))
        return tf.cast(tf.complex(y_out_real, y_out_imag), dtype=inputs.dtype)

    def _save_tensorboard_weight(self, weight_summary, step):
        # No tensorboard things to save
        return None

    def get_description(self):
        return "Complex Dropout:\n\trate={}".format(self.rate)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return Dropout(rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)

    def get_real_equivalent(self):
        return self.__deepcopy__()      # Dropout layer is dtype agnostic

    def trainable_variables(self):
        return []


t_layers_shape = Union[ndarray, List[ComplexLayer], Set[ComplexLayer]]

if __name__ == "__main__":
    import pdb; pdb.set_trace()


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.28'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
