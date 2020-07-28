import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from cvnn import logger
import sys
from typing import Union, List, Set, Tuple
from pdb import set_trace
# Initializers:
# https://www.tensorflow.org/api_docs/python/tf/keras/initializers
# https://keras.io/initializers/


class _RandomGenerator(object):
    """
    Random generator that selects appropriate random ops.
    https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/init_ops_v2.py#L1041
    """

    def __init__(self, seed=None):
        super(_RandomGenerator, self).__init__()
        if seed is not None:
            # Stateless random ops requires 2-int seed.
            self.seed = [seed, 0]
        else:
            self.seed = None

    def random_normal(self, shape, mean=0.0, stddev=1, dtype=tf.dtypes.float32):
        """A deterministic random normal if seed is passed."""
        if self.seed:
            op = stateless_random_ops.stateless_random_normal
        else:
            op = random_ops.random_normal
        return op(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

    def random_uniform(self, shape, minval, maxval, dtype):
        """A deterministic random uniform if seed is passed."""
        if self.seed:
            op = stateless_random_ops.stateless_random_uniform
        else:
            op = random_ops.random_uniform
        return op(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=self.seed)

    def truncated_normal(self, shape, mean, stddev, dtype):
        """A deterministic truncated normal if seed is passed."""
        if self.seed:
            op = stateless_random_ops.stateless_truncated_normal
        else:
            op = random_ops.truncated_normal
        return op(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)


class RandomInitializer:
    def __init__(self, distribution="uniform", seed=None):
        if distribution.lower() not in {"uniform", "normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        else:
            self.distribution = distribution.lower()
        self._random_generator = _RandomGenerator(seed)

    @staticmethod
    def dtype_cast(c_dtype):
        assert c_dtype == tf.complex64 or c_dtype == tf.complex128
        return tf.float32 if c_dtype == tf.complex64 else tf.float64

    @staticmethod
    def _compute_fans(shape):
        """
        Taken from https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/init_ops_v2.py#L994
        Computes the number of input and output units for a weight shape.
        Args:
            shape: Integer shape tuple or TF tensor shape.
        Returns:
            A tuple of scalars (fan_in, fan_out).
        """
        if len(shape) < 1:  # Just to avoid errors for constants.
            fan_in = fan_out = 1.
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assuming convolution kernels (2D, 3D, or more).
            # kernel shape: (..., input_depth, depth)
            receptive_field_size = 1.
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        return fan_in, fan_out

    def _call_random_generator(self, shape, arg, dtype):
        if self.distribution == "uniform":
            return self._random_generator.random_uniform(shape=shape, minval=-arg, maxval=arg, dtype=dtype)
        elif self.distribution == "normal":
            # I make this magic number division because that's what tf does on this case
            return self._random_generator.truncated_normal(shape=shape, mean=0.0, stddev=arg / .87962566103423978,
                                                           dtype=dtype)

    def _get_number(self, shape, c_limit, r_limit, dtype=tf.dtypes.complex64):
        c_limit, r_limit = self._verify_limits(c_limit, r_limit)
        if dtype == np.complex64 or dtype == np.complex128:  # Complex layer
            r_dtype = self.dtype_cast(dtype)
            # TODO: I do not yet understand the random_generator thing. I could use tf.random.uniform once I do
            ret = tf.complex(
                self._call_random_generator(shape=shape, arg=c_limit[0], dtype=r_dtype),
                self._call_random_generator(shape=shape, arg=c_limit[1], dtype=r_dtype))
        elif dtype == np.float32 or dtype == np.float64:
            # ret = tf.random.uniform(shape=shape, minval=-limit, maxval=limit, dtype=dtype, seed=seed, name=name)
            ret = self._call_random_generator(shape=shape, arg=r_limit, dtype=dtype)
        else:
            logger.error("Input_dtype not supported.", exc_info=True)
            sys.exit(-1)
        return ret

    @staticmethod
    def _verify_limits(c_limit, r_limit):
        # TODO: For the moment I do no check. I assume everything is Ok
        return c_limit, r_limit


class GlorotUniform(RandomInitializer):

    def __init__(self, seed):
        super(GlorotUniform, self).__init__(distribution="uniform", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(3. / (fan_in + fan_out)), tf.math.sqrt(3. / (fan_in + fan_out))]
        r_limit = tf.math.sqrt(6. / (fan_in + fan_out))
        return self._get_number(shape, c_limit, r_limit, dtype)


class GlorotUniformCompromise(RandomInitializer):

    def __init__(self, seed):
        super(GlorotUniformCompromise, self).__init__(distribution="uniform", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(3. / (2. * fan_in)), tf.math.sqrt(3. / (2. * fan_out))]
        r_limit = tf.math.sqrt(6. / (fan_in + fan_out))
        return self._get_number(shape, c_limit, r_limit, dtype)


class GlorotNormal(RandomInitializer):

    def __init__(self, seed):
        super(GlorotNormal, self).__init__(distribution="normal", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(1. / (fan_in + fan_out)), tf.math.sqrt(1. / (fan_in + fan_out))]
        r_limit = tf.math.sqrt(2. / (fan_in + fan_out))
        return self._get_number(shape, c_limit, r_limit, dtype)


class HeNormal(RandomInitializer):

    def __init__(self, seed):
        super(HeNormal, self).__init__(distribution="normal", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(1. / fan_in), tf.math.sqrt(1. / fan_in)]       # TODO: Verify this is the eq for complex
        r_limit = tf.math.sqrt(2. / fan_in)
        return self._get_number(shape, c_limit, r_limit, dtype)


class HeUniform(RandomInitializer):

    def __init__(self, seed):
        super(HeUniform, self).__init__(distribution="uniform", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(3. / fan_in), tf.math.sqrt(3. / fan_in)]
        r_limit = tf.math.sqrt(6. / fan_in)
        return self._get_number(shape, c_limit, r_limit, dtype)


class Zeros:
    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.zeros(shape, dtype=dtype)


if __name__ == '__main__':
    # Nothing yet
    print("Cuack")

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.3'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
