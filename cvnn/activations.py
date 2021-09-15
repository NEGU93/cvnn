import tensorflow as tf
from tensorflow.keras.layers import Activation
from typing import Union, Callable, Optional
from tensorflow import Tensor
from numpy import pi

"""
This module contains many complex-valued activation functions to be used by CVNN class.
"""

# logger = logging.getLogger(cvnn.__name__)
t_activation = Union[str, Callable]  # TODO: define better


# Regression
def linear(z: Tensor) -> Tensor:
    """
    Does not apply any activation function. It just outputs the input.
    :param z: Input tensor variable
    :return: z
    """
    return z


def modrelu(z: Tensor, b: float = 1., c: float = 1e-3) -> Tensor:
    """
    mod ReLU presented in "Unitary Evolution Recurrent Neural Networks"
        from M. Arjovsky et al. (2016)
        URL: https://arxiv.org/abs/1511.06464
    A variation of the ReLU named modReLU. It is a pointwise nonlinearity,
    modReLU(z) : C -> C, which affects only the absolute
    value of a complex number, defined:
        modReLU(z) = ReLU(|z|+b)*z/|z|
    TODO: See how to check the non zero abs.
    """
    abs_z = tf.math.abs(z)
    return tf.cast(tf.keras.activations.relu(abs_z + b), dtype=z.dtype) * z / tf.cast(abs_z + c, dtype=z.dtype)


def zrelu(z: Tensor, epsilon=1e-7) -> Tensor:
    """
    zReLU presented in "On Complex Valued Convolutional Neural Networks"
        from Nitzan Guberman (2016).
    This methods let's the output as the input if both real and imaginary parts are positive.

    https://stackoverflow.com/questions/49412717/advanced-custom-activation-function-in-keras-tensorflow
    """
    imag_relu = tf.nn.relu(tf.math.imag(z))
    real_relu = tf.nn.relu(tf.math.real(z))
    ret_real = imag_relu*real_relu / (imag_relu + epsilon)
    ret_imag = imag_relu*real_relu / (real_relu + epsilon)
    ret_val = tf.complex(ret_real, ret_imag)
    return ret_val


def crelu(z: Tensor, alpha: float = 0.0, max_value: Optional[float] = None, threshold: float = 0) -> Tensor:
    """
    Mirror of cart_relu
    """
    return cart_relu(z, alpha, max_value, threshold)


def complex_cardioid(z: Tensor) -> Tensor:
    """
    Complex cardioid presented in "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting"
        from V. Patrick (2017).
        
    This function maintains the phase information while attenuating the magnitude based on the phase itself. 
    For real-valued inputs, it reduces to the ReLU.
    """
    return tf.cast(1 + tf.math.cos(tf.math.angle(z)), dtype=z.dtype) * z / 2.

            
"""
Complex input, real output
"""


def cast_to_real(z: Tensor) -> Tensor:
    return tf.cast(z, z.dtype.real_dtype)


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
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    The softmax of x is calculated by exp(x)/tf.reduce_sum(exp(x)).
        https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
    :param z: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if z.dtype.is_complex:
        return 0.5 * (tf.keras.activations.softmax(tf.math.real(z), axis) + tf.keras.activations.softmax(
            tf.math.real(z), axis))
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


def softmax_real_by_parameter(z: Tensor, axis=-1, params: Optional[dict] = None) -> Tensor:
    if params is None:
        params = {
            'abs': True,
            'angle': True,
            'real': True,
            'imag': True
        }
    result = []
    for k, v in params:
        if k == 'abs' and v:
            result.append(tf.keras.activations.softmax(tf.math.abs(z), axis))
        if k == 'angle' and v:
            result.append(tf.keras.activations.softmax(tf.math.angle(z), axis))
        if k == 'real' and v:
            result.append(tf.keras.activations.softmax(tf.math.real(z), axis))
        if k == 'imag' and v:
            result.append(tf.keras.activations.softmax(tf.math.imag(z), axis))
    return tf.convert_to_tensor(result)


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


def softmax_real_with_polar(z: Tensor, axis=-1) -> Tensor:
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
        return 0.5 * (tf.keras.activations.softmax(tf.math.abs(z), axis) + tf.keras.activations.softmax(tf.math.angle(z),
                                                                                                        axis))
    else:
        return tf.keras.activations.softmax(z, axis)


"""
etf Functions
"""


def etf_circular_tan(z: Tensor) -> Tensor:
    return tf.math.tan(z)


def etf_circular_sin(z: Tensor) -> Tensor:
    return tf.math.sin(z)


def etf_inv_circular_atan(z: Tensor) -> Tensor:
    return tf.math.atan(z)


def etf_inv_circular_asin(z: Tensor) -> Tensor:
    return tf.math.asin(z)


def etf_inv_circular_acos(z: Tensor) -> Tensor:
    return tf.math.acos(z)


def etf_circular_tanh(z: Tensor) -> Tensor:
    return tf.math.tanh(z)


def etf_circular_sinh(z: Tensor) -> Tensor:
    return tf.math.sinh(z)


def etf_inv_circular_atanh(z: Tensor) -> Tensor:
    return tf.math.atanh(z)


def etf_inv_circular_asinh(z: Tensor) -> Tensor:
    return tf.math.asinh(z)


"""
Phasor Networks
"""


def georgiou_cdbp(z:Tensor, r: float = 1, c: float = 1e-3) -> Tensor:
    """
    Activation function proposed by G. M. Georgioy and C. Koutsougeras in
        https://ieeexplore.ieee.org/abstract/document/142037
    """
    return z / tf.cast(c + tf.math.abs(z)/r, dtype=z.dtype)


def complex_signum(z: Tensor, k: Optional[int] = None) -> Tensor:
    """
    Complex signum activation function is very similar to mvn_activation.
    For a detailed explanation refer to:
        https://ieeexplore.ieee.org/abstract/document/548176
    """
    if k:
        # values = np.linspace(pi / k, 2 * pi - pi / k, k)
        angle_cast = tf.math.floor(tf.math.angle(z) * k / (2 * pi))
        # import pdb; pdb.set_trace()
        return tf.math.exp(tf.complex(
            tf.zeros(tf.shape(z), dtype=z.dtype.real_dtype), angle_cast * 2 * pi / k))
    else:
        return tf.math.exp(tf.complex(tf.zeros(tf.shape(z), dtype=z.dtype.real_dtype), tf.math.angle(z)))


def mvn_activation(z: Tensor, k: Optional[int] = None) -> Tensor:
    """
    Function inspired by Naum Aizenberg.
        A multi-valued neuron (MVN) is a neural element with n inputs and one output lying on the unit circle,
        and with complex-valued weights.
    Works:
        https://link.springer.com/article/10.1007%2FBF01068667
        http://pefmath2.etf.rs/files/93/399.pdf
    """
    if k:
        # values = np.linspace(pi / k, 2 * pi - pi / k, k)
        angle_cast = tf.math.floor(tf.math.angle(z) * k / (2 * pi))
        # import pdb; pdb.set_trace()
        return tf.math.exp(tf.complex(
            tf.zeros(tf.shape(z), dtype=z.dtype.real_dtype), (angle_cast + 0.5) * 2 * pi / k))
    else:
        return tf.math.exp(tf.complex(tf.zeros(tf.shape(z), dtype=z.dtype.real_dtype), tf.math.angle(z)))


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
                              tf.keras.activations.sigmoid(tf.math.imag(z))), 
                   dtype=z.dtype)


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
    'linear': linear,
    # Complex input, real output
    'cast_to_real': cast_to_real,
    'convert_to_real_with_abs': convert_to_real_with_abs,
    'sigmoid_real': sigmoid_real,
    'softmax_real_with_abs': softmax_real_with_abs,
    'softmax_real_with_avg': softmax_real_with_avg,
    'softmax_real_with_mult': softmax_real_with_mult,
    'softmax_of_softmax_real_with_mult': softmax_of_softmax_real_with_mult,
    'softmax_of_softmax_real_with_avg': softmax_of_softmax_real_with_avg,
    'softmax_real_with_polar': softmax_real_with_polar,
    # Phasor networks
    'georgiou_cdbp': georgiou_cdbp,
    'mvn_activation': mvn_activation,
    'complex_signum': complex_signum,
    # Type A (cartesian)
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
    # Type B (polar)
    'pol_tanh': pol_tanh,
    'pol_sigmoid': pol_sigmoid,
    'pol_selu': pol_selu,
    # Elementary Transcendental Functions (ETF)
    'etf_circular_tan': etf_circular_tan,
    'etf_circular_sin': etf_circular_sin,
    'etf_inv_circular_atan': etf_inv_circular_atan,
    'etf_inv_circular_asin': etf_inv_circular_asin,
    'etf_inv_circular_acos': etf_inv_circular_acos,
    'etf_circular_tanh': etf_circular_tanh,
    'etf_circular_sinh': etf_circular_sinh,
    'etf_inv_circular_atanh': etf_inv_circular_atanh,
    'etf_inv_circular_asinh': etf_inv_circular_asinh,
    # ReLU
    'modrelu': modrelu,
    'crelu': crelu,
    'zrelu': zrelu,
    'complex_cardioid': complex_cardioid
}

if __name__ == '__main__':
    x = tf.constant([-2, 1.0, 0.0, 1.0, -3, 0.8, 0.1], dtype=tf.float32)
    y = tf.constant([-2.5, -1.5, 0.0, 1.0, 2, 0.4, -0.4], dtype=tf.float32)
    z = tf.complex(x, y)
    result = crelu(z)
    result = modrelu(z, 4)
    result = zrelu(z)
    result = complex_cardioid(z)
    """import matplotlib.pyplot as plt
    import numpy as np

    x = tf.constant([-2, 1.0, 0.0, 1.0, -3, 0.8, 0.1], dtype=tf.float32)
    y = tf.constant([-2.5, -1.5, 0.0, 1.0, 2, 0.4, -0.4], dtype=tf.float32)
    z = tf.complex(x, y)
    result = georgiou_cdbp(z)

    ax = plt.axes()
    ax.scatter(tf.math.real(z), tf.math.imag(z), color='red')
    ax.scatter(tf.math.real(result), tf.math.imag(result), color='blue')
    for x, y, dx, dy in zip(tf.math.real(z), tf.math.imag(z),
                            tf.math.real(result) - tf.math.real(z),
                            tf.math.imag(result) - tf.math.imag(z)):
        ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.1)
    t = np.linspace(0, np.pi * 2, 100)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)

    yabs_max = abs(max(ax.get_ylim(), key=abs))
    xabs_max = abs(max(ax.get_xlim(), key=abs))
    axis_max = max(yabs_max, xabs_max)

    ax.set_ylim(ymin=-axis_max, ymax=axis_max)
    ax.set_xlim(xmin=-axis_max, xmax=axis_max)
    plt.show()"""
    

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.21'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
