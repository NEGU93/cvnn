import tensorflow as tf

"""
This module contains many complex-valued activation functions to be used by CVNN class.
"""

"""
TYPE A: Cartesian form.
"""


# Regression
def act_linear(z):
    """
    Does not apply any activation function. It just outputs the input.
    :param z: Input tensor variable
    :return: z
    """
    return z


def act_cart_sigmoid(z):
    """
    Applies the function (1.0 / (1.0 + exp(-x))) + j * (1.0 / (1.0 + exp(-y))) where z = x + j * y
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid
    :param z: Tensor to be used as input of the activation function
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.sigmoid(tf.math.real(z)), tf.keras.activations.sigmoid(tf.math.imag(z)))


def act_cart_elu(z, alpha=1.0):
    """
    Applies the "Exponential linear unit": x if x > 0 and alpha * (exp(x)-1) if x < 0
    To both the real and imaginary part of z.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu
    :param z: Input tensor.
    :param alpha: A scalar, slope of negative section.
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.elu(tf.math.real(z, alpha)),
                      tf.keras.activations.elu(tf.math.imag(z, alpha)))


def act_cart_exponential(z):
    """
    Exponential activation function. Applies to both the real and imag part of z the exponential activation: exp(x)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.exponential(tf.math.real(z)),
                      tf.keras.activations.exponential(tf.math.imag(z)))


def act_cart_hard_sigmoid(z):
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
    return tf.complex(tf.keras.activations.hard_sigmoid(tf.math.real(z)),
                      tf.keras.activations.hard_sigmoid(tf.math.imag(z)))


def act_cart_relu(z, alpha=0.0, max_value=None, threshold=0):
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
    return tf.complex(tf.keras.activations.relu(tf.math.real(z, alpha, max_value, threshold)),
                      tf.keras.activations.relu(tf.math.imag(z, alpha, max_value, threshold)))


def act_cart_selu(z):
    """
    Applies Scaled Exponential Linear Unit (SELU) to both the real and imag part of z.
    The scaled exponential unit activation: scale * elu(x, alpha).
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.selu(tf.math.real(z)), tf.keras.activations.selu(tf.math.imag(z)))


def act_cart_softplus(z):
    """
    Applies Softplus activation function to both the real and imag part of z.
    The Softplus function: log(exp(x) + 1)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.softplus(tf.math.real(z)), tf.keras.activations.softplus(tf.math.imag(z)))


def act_cart_softsign(z):
    """
    Applies Softsign activation function to both the real and imag part of z.
    The softsign activation: x / (abs(x) + 1).      TODO: typo in tensorflow references (softplus instead of softsign)
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.softsign(tf.math.real(z)), tf.keras.activations.softsign(tf.math.imag(z)))


def act_cart_tanh(z):
    """
    Applies Hyperbolic Tangent (tanh) activation function to both the real and imag part of z.
    The tanh activation: tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x))).
    The derivative if tanh is computed as 1 - tanh^2 so it should be fast to compute for backprop.
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh
    :param z: Input tensor.
    :return: Tensor result of the applied activation function
    """
    return tf.complex(tf.keras.activations.tanh(tf.math.real(z)), tf.keras.activations.tanh(tf.math.imag(z)))


# Classification
def act_cart_softmax(z, axis=-1):
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
    return tf.complex(tf.keras.activations.softmax(tf.math.real(z, axis)),
                      tf.keras.activations.softmax(tf.math.imag(z, axis)))

"""
TYPE B: Polar form.
"""