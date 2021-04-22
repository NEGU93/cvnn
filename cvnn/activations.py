import logging
import tensorflow as tf
from tensorflow.keras.layers import Activation
from typing import Union, Callable, Optional
from tensorflow import Tensor
import cvnn

"""
This module contains many complex-valued activation functions to be used by CVNN class.
"""

logger = logging.getLogger(cvnn.__name__)
t_activation = Union[str, Callable]  # TODO: define better


# Regression
def linear(z: Tensor) -> Tensor:
    """
    Does not apply any activation function. It just outputs the input.
    :param z: Input tensor variable
    :return: z
    """
    return z


"""
Complex input, real output
"""


def sigmoid_real(z: Tensor) -> Tensor:
    return tf.keras.activations.sigmoid(tf.math.real(z) + tf.math.imag(z))


def softmax_real_with_abs(z: Tensor, axis=-1) -> Tensor:
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
    if z.dtype.is_complex:
        return tf.keras.activations.softmax(tf.math.abs(z), axis)
    else:
        return tf.keras.activations.softmax(z, axis)


def softmax_real_with_avg(z: Tensor, axis=-1) -> Tensor:
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
    if z.dtype.is_complex:
        return tf.keras.activations.softmax(tf.math.real(z), axis) + tf.keras.activations.softmax(tf.math.real(z), axis)
    else:
        return tf.keras.activations.softmax(z, axis)


def softmax_real_with_mult(z: Tensor, axis=-1) -> Tensor:
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
    if z.dtype.is_complex:
        return tf.keras.activations.softmax(tf.math.real(z), axis) * tf.keras.activations.softmax(tf.math.real(z), axis)
    else:
        return tf.keras.activations.softmax(z, axis)


def softmax_of_softmax_real_with_mult(z: Tensor, axis=-1) -> Tensor:
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
    if z.dtype.is_complex:
        return tf.keras.activations.softmax(
            tf.keras.activations.softmax(tf.math.real(z), axis) * tf.keras.activations.softmax(tf.math.real(z), axis),
            axis)
    else:
        return tf.keras.activations.softmax(z, axis)


def softmax_of_softmax_real_with_avg(z: Tensor, axis=-1) -> Tensor:
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
    if z.dtype.is_complex:
        return tf.keras.activations.softmax(
            tf.keras.activations.softmax(tf.math.real(z), axis) + tf.keras.activations.softmax(tf.math.real(z), axis),
            axis)
    else:
        return tf.keras.activations.softmax(z, axis)


def convert_to_real_with_abs(z: Tensor) -> Tensor:
    """
    Applies the absolute value and returns a real-valued output.
    :param z: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if z.dtype.is_complex:
        return tf.math.abs(z)
    else:
        return z


"""
TYPE A: Cartesian form.
"""
# TODO: shall I use tf.nn or tf.keras.activation modules?
# https://stackoverflow.com/questions/54761088/tf-nn-relu-vs-tf-keras-activations-relu
# nn has leaky relu, activation doesn't


def cart_sigmoid(z: Tensor) -> Tensor:
    """
    Applies the function (1.0 / (1.0 + exp(-x))) + j * (1.0 / (1.0 + exp(-y))) where z = x + j * y
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid
    :param z: Tensor to be used as input of the activation function
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.sigmoid(tf.math.real(z)),
                              tf.keras.activations.sigmoid(tf.math.imag(z))), dtype=z.dtype)


def cart_elu(z: Tensor, alpha=1.0) -> Tensor:
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


def cart_exponential(z: Tensor) -> Tensor:
    """
    Exponential activation function. Applies to both the real and imag part of z the exponential activation: exp(x)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.exponential(tf.math.real(z)),
                              tf.keras.activations.exponential(tf.math.imag(z))), dtype=z.dtype)


def cart_hard_sigmoid(z: Tensor) -> Tensor:
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


def cart_relu(z: Tensor, alpha: float = 0.0, max_value: Optional[float] = None, threshold: float = 0) -> Tensor:
    """
    Applies Rectified Linear Unit to both the real and imag part of z
    The relu function, with default values, it returns element-wise max(x, 0).
    Otherwise, it follows:  f(x) = max_value for x >= max_value,
                            f(x) = x for threshold <= x < max_value,
                            f(x) = alpha * (x - threshold) otherwise.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu
    :param z: Tensor -- Input tensor.
    :param alpha: float -- A float that governs the slope for values lower than the threshold (default 0.0).
    :param max_value: Optional float -- A float that sets the saturation threshold (the largest value the function will return)
        (default None).
    :param threshold: float -- A float giving the threshold value of the activation function below which
        values will be damped or set to zero (default 0).
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.relu(tf.math.real(z), alpha, max_value, threshold),
                              tf.keras.activations.relu(tf.math.imag(z), alpha, max_value, threshold)), dtype=z.dtype)


def cart_leaky_relu(z: Tensor, alpha=0.2, name=None) -> Tensor:
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


def cart_selu(z: Tensor) -> Tensor:
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


def cart_softplus(z: Tensor) -> Tensor:
    """
    Applies Softplus activation function to both the real and imag part of z.
    The Softplus function: log(exp(x) + 1)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.softplus(tf.math.real(z)),
                              tf.keras.activations.softplus(tf.math.imag(z))), dtype=z.dtype)


def cart_softsign(z: Tensor) -> Tensor:
    """
    Applies Softsign activation function to both the real and imag part of z.
    The softsign activation: x / (abs(x) + 1).      TODO: typo in tensorflow references (softplus instead of softsign)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.activations.softsign(tf.math.real(z)),
                              tf.keras.activations.softsign(tf.math.imag(z))), dtype=z.dtype)


def cart_tanh(z: Tensor) -> Tensor:
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
def cart_softmax(z: Tensor, axis=-1) -> Tensor:
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
# For all ReLU functions, the polar form makes no real sense. If we keep the phase because abs(z) > 0


def _apply_pol(z: Tensor, amp_fun: Callable[[Tensor], Tensor],
               pha_fun: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
    amp = amp_fun(tf.math.abs(z))
    pha = tf.math.angle(z)
    if pha_fun is not None:
        pha = pha_fun(pha)
    return tf.cast(tf.complex(amp * tf.math.cos(pha), amp * tf.math.sin(pha)), dtype=z.dtype)


def pol_tanh(z: Tensor) -> Tensor:
    """
    Applies Hyperbolic Tangent (tanh) activation function to the amplitude of the complex number
        leaving the phase untouched.
    The derivative if tanh is computed as 1 - tanh^2 so it should be fast to compute for backprop.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return _apply_pol(z, tf.keras.activations.tanh)


def pol_sigmoid(z: Tensor) -> Tensor:
    """
    Applies the sigmoid function to the amplitude of the complex number leaving the phase untouched
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid
    :param z: Tensor to be used as input of the activation function
    :return: Tensor result of the applied activation function
    """
    return _apply_pol(z, tf.keras.activations.sigmoid)


def pol_selu(z: Tensor) -> Tensor:
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
    'linear': Activation(linear),
    'convert_to_real_with_abs': Activation(convert_to_real_with_abs),
    'sigmoid_real': Activation(sigmoid_real),
    'softmax_real_with_abs': Activation(softmax_real_with_abs),
    'softmax_real_with_avg': Activation(softmax_real_with_avg),
    'softmax_real_with_mult': Activation(softmax_real_with_mult),
    'softmax_of_softmax_real_with_mult': Activation(softmax_of_softmax_real_with_mult),
    'softmax_of_softmax_real_with_avg': Activation(softmax_of_softmax_real_with_avg),
    'cart_sigmoid': Activation(cart_sigmoid),
    'cart_elu': Activation(cart_elu),
    'cart_exponential': Activation(cart_exponential),
    'cart_hard_sigmoid': Activation(cart_hard_sigmoid),
    'cart_relu': Activation(cart_relu),
    'cart_leaky_relu': Activation(cart_leaky_relu),
    'cart_selu': Activation(cart_selu),
    'cart_softplus': Activation(cart_softplus),
    'cart_softsign': Activation(cart_softsign),
    'cart_tanh': Activation(cart_tanh),
    'cart_softmax': Activation(cart_softmax),
    'pol_tanh': Activation(pol_tanh),
    'pol_sigmoid': Activation(pol_sigmoid),
    'pol_selu': Activation(pol_selu)
}


__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.10'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
