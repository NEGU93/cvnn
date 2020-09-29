import tensorflow as tf
import sys
import cvnn
import logging

"""
This module contains many complex-valued activation functions to be used by CVNN class.
"""

logger = logging.getLogger(cvnn.__name__)


def apply_activation(act_fun, out):
    """
    Applies activation function `act` to variable `out`
    :param out: Tensor to whom the activation function will be applied
    :param act_fun: function to be applied to out. See the list fo possible activation functions on:
        https://complex-valued-neural-networks.readthedocs.io/en/latest/act_fun.html
    :return: Tensor with the applied activation function
    """
    if act_fun is None:  # No activation function declared
        return out
    elif callable(act_fun):
        if act_fun.__module__ == 'activation_functions' or \
                act_fun.__module__ == 'tensorflow.python.keras.activations':
            return act_fun(out)  # TODO: for the moment is not be possible to give parameters like alpha
        else:
            logger.error("apply_activation Unknown activation function.\n\t"
                         "Can only use activations declared on activation_functions.py or keras.activations")
            sys.exit(-1)
    elif isinstance(act_fun, str):
        try:
            return act_dispatcher[act_fun.lower()](out)
        except KeyError:
            logger.warning(str(act_fun) + " is not callable, ignoring it")
        return out


def softmax_real(z, axis=-1):
    """
    Applies the softmax function to the modulus of z.
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    The softmax of x is calculated by exp(x)/tf.reduce_sum(exp(x)).
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    :param z: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    return tf.keras.activations.softmax(tf.math.abs(z), axis)


# Regression
def linear(z):
    """
    Does not apply any activation function. It just outputs the input.
    :param z: Input tensor variable
    :return: z
    """
    return z


"""
TYPE A: Cartesian form.
"""

# TODO: shall I use tf.nn or tf.keras.activation modules?
# https://stackoverflow.com/questions/54761088/tf-nn-relu-vs-tf-keras-activations-relu
# nn has leaky relu, activation doesn't


def cart_sigmoid(z):
    """
    Applies the function (1.0 / (1.0 + exp(-x))) + j * (1.0 / (1.0 + exp(-y))) where z = x + j * y
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid
    :param z: Tensor to be used as input of the activation function
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.sigmoid(tf.math.real(z)),
                              tf.keras.activations.sigmoid(tf.math.imag(z))), dtype=z.dtype)


def cart_elu(z, alpha=1.0):
    """
    Applies the "Exponential linear unit": x if x > 0 and alpha * (exp(x)-1) if x < 0
    To both the real and imaginary part of z.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu
    :param z: Input tensor.
    :param alpha: A scalar, slope of negative section.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.elu(tf.math.real(z), alpha),
                              tf.keras.activations.elu(tf.math.imag(z), alpha)), dtype=z.dtype)


def cart_exponential(z):
    """
    Exponential activation function. Applies to both the real and imag part of z the exponential activation: exp(x)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.exponential(tf.math.real(z)),
                              tf.keras.activations.exponential(tf.math.imag(z))), dtype=z.dtype)


def cart_hard_sigmoid(z):
    """
    Applies the Hard Sigmoid function to both the real and imag part of z.
    The hard sigmoid function is faster to compute than sigmoid activation.
    Hard sigmoid activation:    0               if x < -2.5
                                1               if x > 2.5
                                0.2 * x + 0.5   if -2.5 <= x <= 2.5
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.hard_sigmoid(tf.math.real(z)),
                              tf.keras.activations.hard_sigmoid(tf.math.imag(z))), dtype=z.dtype)


def cart_relu(z, alpha=0.0, max_value=None, threshold=0):
    """
    Applies Rectified Linear Unit to both the real and imag part of z
    The relu function, with default values, it returns element-wise max(x, 0).
    Otherwise, it follows:  f(x) = max_value for x >= max_value,
                            f(x) = x for threshold <= x < max_value,
                            f(x) = alpha * (x - threshold) otherwise.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.relu(tf.math.real(z), alpha, max_value, threshold),
                              tf.keras.activations.relu(tf.math.imag(z), alpha, max_value, threshold)), dtype=z.dtype)


def cart_leaky_relu(z, alpha=0.2, name=None):
    """
    Applies Leaky Rectified Linear Unit to both the real and imag part of z
    https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
    http://robotics.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    :param z: Input tensor.
    :param alpha: Slope of the activation function at x < 0. Default: 0.2
    :param name: A name for the operation (optional).
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.nn.leaky_relu(tf.math.real(z), alpha, name),
                              tf.nn.leaky_relu(tf.math.imag(z), alpha, name)), dtype=z.dtype)


def cart_selu(z):
    """
    Applies Scaled Exponential Linear Unit (SELU) to both the real and imag part of z.
    The scaled exponential unit activation: scale * elu(x, alpha).
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu
    https://arxiv.org/abs/1706.02515
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.selu(tf.math.real(z)),
                              tf.keras.activations.selu(tf.math.imag(z))), dtype=z.dtype)


def cart_softplus(z):
    """
    Applies Softplus activation function to both the real and imag part of z.
    The Softplus function: log(exp(x) + 1)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.softplus(tf.math.real(z)),
                              tf.keras.activations.softplus(tf.math.imag(z))), dtype=z.dtype)


def cart_softsign(z):
    """
    Applies Softsign activation function to both the real and imag part of z.
    The softsign activation: x / (abs(x) + 1).      TODO: typo in tensorflow references (softplus instead of softsign)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.softsign(tf.math.real(z)),
                              tf.keras.activations.softsign(tf.math.imag(z))), dtype=z.dtype)


def cart_tanh(z):
    """
    Applies Hyperbolic Tangent (tanh) activation function to both the real and imag part of z.
    The tanh activation: tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x))).
    The derivative if tanh is computed as 1 - tanh^2 so it should be fast to compute for backprop.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.tanh(tf.math.real(z)),
                              tf.keras.activations.tanh(tf.math.imag(z))), dtype=z.dtype)


# Classification
def cart_softmax(z, axis=-1):
    """
    Applies the softmax function to both the real and imag part of z.
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    The softmax of x is calculated by exp(x)/tf.reduce_sum(exp(x)).
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.softmax(tf.math.real(z), axis),
                              tf.keras.activations.softmax(tf.math.imag(z), axis)), dtype=z.dtype)


"""
TYPE B: Polar form.
"""
# TODO: for all ReLU functions, the polar form makes no real sense. If we keep the phase because abs(z) > 0


def pol_selu(z):
    """
    Applies Scaled Exponential Linear Unit (SELU) to the absolute value of z, keeping the phase unchanged.
    The scaled exponential unit activation: scale * elu(x, alpha).
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu
    https://arxiv.org/abs/1706.02515
    :param z: Input tensor.
    :return: Tensor result of the applied activation function

    Logic:
        I must mantain the phase (angle) so: cos(theta) = x_0/r_0 = x_1/r_1.
        For real case, x_0 = r_0 so it also works.
    """
    r_0 = tf.abs(z)
    r_1 = tf.keras.activations.selu(r_0)
    return tf.cast(tf.complex(tf.math.real(z) * r_1 / r_0, tf.math.imag(z) * r_1 / r_0), dtype=z.dtype)


act_dispatcher = {
    'linear': linear,
    'cart_sigmoid': cart_sigmoid,
    'cart_elu': cart_elu,
    'cart_exponential': cart_exponential,
    'cart_hard_sigmoid': cart_hard_sigmoid,
    'cart_relu': cart_relu,
    'cart_leaky_relu': cart_leaky_relu,
    'cart_selu': cart_selu,
    'cart_softplus': cart_softplus,
    'cart_softsign': cart_softsign,
    'cart_tanh': cart_tanh,
    'cart_softmax': cart_softmax,
    'softmax_real': softmax_real,
    'pol_selu': pol_selu
}


__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.7'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
