from abc import ABC, abstractmethod
import tensorflow as tf
import sys
import cvnn.activation_functions as act
from cvnn.utils import get_func_name
import numpy as np

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

supported_dtypes = (np.complex64, np.float32)


class Layer(ABC):
    def __init__(self, input_size, output_size, layer_number, activation=None,
                 input_dtype=np.complex64, output_dtype=np.complex64):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_number = layer_number
        self.activation = activation
        if output_dtype == np.complex64 and input_dtype == np.float32:
            # TODO: can't it?
            sys.exit("Layer::__init__: if input dtype is real output cannot be complex")
        if input_dtype not in supported_dtypes:
            sys.exit("ERROR:Layer::__init__: unsupported input_dtype " + str(input_dtype))
        self.input_dtype = input_dtype
        if output_dtype not in supported_dtypes:
            sys.exit("ERROR:Layer::__init__: unsupported output_dtype " + str(output_dtype))
        self.output_dtype = output_dtype
        super().__init__()

    @staticmethod
    def _apply_activation(act_fun, out):
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
                sys.exit("Cvnn::_apply_activation Unknown activation function.\n\t "
                         "Can only use activations declared on activation_functions.py or keras.activations")
        elif isinstance(act_fun, str):
            try:
                return act_dispatcher[act_fun](out)
            except KeyError:
                print("WARNING: Cvnn::_apply_function: " + str(act_fun) + " is not callable, ignoring it")
            return out

    @abstractmethod
    def apply_layer(self, input, output_options):
        pass

    @abstractmethod
    def get_description(self):
        pass


class Dense(Layer):
    def apply_layer(self, input, output_options):
        # TODO: treat bias as a weight. It might optimize training (no add operation, only mult)
        with tf.compat.v1.name_scope("dense_layer_" + str(self.layer_number)):
            # Create weight matrix initialized randomely from N~(0, 0.01)
            # TODO: be able to choose the initializer
            if self.input_dtype == np.complex64:    # Complex layer
                w = tf.Variable(
                    tf.complex(tf.keras.initializers.GlorotUniform()(shape=(self.input_size, self.output_size)),
                               tf.keras.initializers.GlorotUniform()(shape=(self.input_size, self.output_size))),
                    name="weights" + str(self.layer_number))
                b = tf.Variable(tf.complex(tf.zeros(self.output_size),
                                tf.zeros(self.output_size)), name="bias" + str(self.layer_number))
            elif self.input_dtype == np.float32:       # Real Layer
                w = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(self.input_size, self.output_size)),
                                name="weights" + str(self.layer_number))
                b = tf.Variable(tf.zeros(self.output_size), name="bias" + str(self.layer_number))
            else:
                # This case should never happen. The constructor should already have checked this
                sys.exit("ERROR: Dense::apply_layer: input_dtype not supported.")
            if output_options.tensorboard:
                tf.compat.v1.summary.histogram('real_weight_' + str(self.layer_number), tf.math.real(w))
                tf.compat.v1.summary.histogram('imag_weight_' + str(self.layer_number), tf.math.imag(w))
                tf.compat.v1.summary.histogram('real_bias_' + str(self.layer_number), tf.math.real(b))
                tf.compat.v1.summary.histogram('imag_bias_' + str(self.layer_number), tf.math.imag(b))
            out = tf.add(tf.matmul(input, w), b)

            if tf.dtypes.as_dtype(np.dtype(self.output_dtype)) != out.dtype:  # Case for real output / real labels
                assert self.output_dtype == np.float32 and out.dtype == tf.dtypes.complex64
                print("WARNING:Dense::apply_layer: Automatically casting output as abs because output should be real")
                out = tf.abs(out)  # TODO: Shall I do abs or what?
            return self._apply_activation(self.activation, out), [w, b]

    def get_description(self):
        fun_name = get_func_name(self.activation)
        out_str = "Dense layer: output size = " + str(self.output_size) + "; act_fun = " + fun_name + "\n"
        return out_str


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.1'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
