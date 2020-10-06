import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from cvnn import logger
import sys
from pdb import set_trace
# Typing
from typing import Optional
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
    """
    Random initializer helps generate a random tensor of:
        - Either complex or real (floating) data type
        - Either Uniform or Normal distribution
        - Zero mean

    How to use it:
        ```
        # creates a complex tensor of shape (3, 3) distribution with
        #   Re{random_tensor} ~ U[-2, 2] and Im{random_tensor} ~ U[-3, 3]
        random_tensor = RandomInitializer(distribution="uniform")(shape=(3, 3), c_limit=[2, 3], dtype=tf.complex)
        ```
    """
    def __init__(self, distribution: str = "uniform", seed: Optional[int] = None):
        """
        :param distribution: It can be either a uniform or a normal distribution.
        :param seed: A Python integer. Used to create a random seed for the distribution. See tf.random.set_seed.
        """
        if distribution.lower() not in {"uniform", "normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        else:
            self.distribution = distribution.lower()
        self._random_generator = _RandomGenerator(seed)     # ATTENTION: I do not check the seed.

    @staticmethod
    def dtype_cast(c_dtype):
        """
        Equivalent to is_floating attribute of `tf.dtypes.Dtype` but casts the input and checks is complex
        (https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)
        """
        c_dtype = tf.as_dtype(c_dtype)
        if not c_dtype.is_complex:
            logger.error("Expecting a complex dtype, got " + str(c_dtype))
            sys.exit(-1)
        return c_dtype.real_dtype

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

    def get_random_tensor(self, shape, c_arg=None, r_arg=None, dtype=tf.dtypes.complex64):
        """
        Outputs random values either uniform or normal according to initialization
        :param shape: The shape of the output tensor.
        :param r_arg: Argument.
            If uniform, the output will be a distribution between [-arg, arg].
            If Normal, the output will be a zero-mean gaussian distribution with arg stddev
        :param c_arg: Tuple of the argument for the real and imaginary part respectively.
            *Note: either c_arg or r_arg will be used according to dtype parameter,
                the other will be ignored and can be None.
        :param dtype: The type of the output. Default tf.complex.
        """
        dtype = tf.as_dtype(dtype)
        c_arg, r_arg = self._verify_limits(c_arg, r_arg, dtype)
        if dtype.is_complex:  # Complex layer
            r_dtype = self.dtype_cast(dtype)
            c_arg = tf.cast(c_arg, r_dtype)
            # TODO: I do not yet understand the random_generator thing. I could use tf.random.uniform once I do
            ret = tf.complex(
                self._call_random_generator(shape=shape, arg=c_arg[0], dtype=r_dtype),
                self._call_random_generator(shape=shape, arg=c_arg[1], dtype=r_dtype))
        elif dtype.is_floating:     # Real Layer
            # ret = tf.random.uniform(shape=shape, minval=-limit, maxval=limit, dtype=dtype, seed=seed, name=name)
            r_arg = tf.cast(r_arg, dtype)
            ret = self._call_random_generator(shape=shape, arg=r_arg, dtype=dtype)
        else:
            logger.error("Input_dtype not supported.", exc_info=True)
            sys.exit(-1)
        return ret

    @staticmethod
    def _verify_limits(c_limit, r_limit, dtype):
        if c_limit is None and r_limit is None:
            logger.error("Either one argument must not None")
            sys.exit(-1)
        elif dtype.is_complex and c_limit is None:
            logger.error("complex argument is None but dtype is complex")
            sys.exit(-1)
        elif dtype.is_floating and r_limit is None:
            logger.error("real argument is None but dtype is real")
            sys.exit(-1)
        # TODO: For the moment I do no check the c_limit size 2 and c/r_limit dtype.
        return c_limit, r_limit


class GlorotUniform(RandomInitializer):
    """
    The Glorot uniform initializer, also called Xavier uniform initializer.
    Reference: http://proceedings.mlr.press/v9/glorot10a.html
    Draws samples from a uniform distribution:
        - Real case: `x ~ U[-limit, limit]` where `limit = sqrt(6 / (fan_in + fan_out))`
        - Complex case: `z / Re{z} = Im{z} ~ U[-limit, limit]` where `limit = sqrt(3 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units.

    ```
    # Standalone usage:
    import cvnn
    initializer = cvnn.initializers.GlorotUniform()
    values = initializer(shape=(2, 2))                  # Returns a complex Glorot Uniform tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.GlorotUniform()
    layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)
    ```
    """

    __name__ = "Glorot Uniform"

    def __init__(self, seed=None, scale=1.):
        """
        :param seed: A Python integer. An initializer created with a given seed will always produce
            the same random tensor for a given shape and dtype.
        :param scale: Default 1. Scales the limit as `limit = scale * limit`
        """
        if isinstance(scale, float):
            assert scale > 0, "scale must be more than 0. Got " + str(scale)
        else:
            logger.error("scale must be a float. Got " + str(scale))
            sys.exit(-1)
        self.scale = scale
        super(GlorotUniform, self).__init__(distribution="uniform", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        """
        Returns a tensor object initialized as specified by the initializer.
        :param shape: Shape of the tensor.
        :param dtype: Optional dtype of the tensor. Either floating or complex. ex: tf.complex64 or tf.float32
        """
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [self.scale * tf.math.sqrt(3. / (fan_in + fan_out)),
                   self.scale * tf.math.sqrt(3. / (fan_in + fan_out))]
        r_limit = self.scale * tf.math.sqrt(6. / (fan_in + fan_out))
        return self.get_random_tensor(shape, c_limit, r_limit, dtype)


class GlorotNormal(RandomInitializer):
    """
    The Glorot normal initializer, also called Xavier normal initializer.
    Reference: http://proceedings.mlr.press/v9/glorot10a.html
        *Note: The reference actually refers to the uniform case but it's analysis was adapted for a normal distribution
    Draws samples from a truncated normal distribution centered on 0 with
     - Real case: `stddev = sqrt(2 / (fan_in + fan_out))`
     - Complex case: real part stddev = complex part stddev = `1 / sqrt(fan_in + fan_out)`
    where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units.

    ```
    # Standalone usage:
    import cvnn
    initializer = cvnn.initializers.GlorotNormal()
    values = initializer(shape=(2, 2))                  # Returns a complex Glorot Normal tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.GlorotNormal()
    layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)
    ```
    """
    __name__ = "Glorot Normal"

    def __init__(self, seed=None):
        super(GlorotNormal, self).__init__(distribution="normal", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        """
        Returns a tensor object initialized as specified by the initializer.
        :param shape: Shape of the tensor.
<<<<<<< HEAD
        :param dtype: Optinal dtype of the tensor. Either floating or complex. ex: tf.complex64 or tf.float32
=======
        :param dtype: Optional dtype of the tensor. Either floating or complex. ex: tf.complex63 or tf.float32
>>>>>>> 2a18681b0f0ec5e3a1787b8dd7287f1f3f0de985
        """
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(1. / (fan_in + fan_out)), tf.math.sqrt(1. / (fan_in + fan_out))]
        r_limit = tf.math.sqrt(2. / (fan_in + fan_out))
        return self.get_random_tensor(shape, c_limit, r_limit, dtype)


class GlorotUniformCompromise(RandomInitializer):
    """
    The Glorot uniform initializer, also called Xavier uniform initializer.
    Reference: http://proceedings.mlr.press/v9/glorot10a.html
    Draws samples from a uniform distribution:
        - Real case: `x ~ U[-limit, limit]` where `limit = sqrt(6 / (fan_in + fan_out))`
        - Complex case: `z /
            - Re{z} ~ U[-limit, limit]` where `limit = sqrt(3 / (2 * fan_in ))`
            - Im{z} ~ U[-limit, limit]` where `limit = sqrt(3 / ( 2 * fan_out))`
    where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units.

    ```
    # Standalone usage:
    import cvnn
    initializer = cvnn.initializers.GlorotUniformCompromise()
    values = initializer(shape=(2, 2))                  # Returns a complex Glorot Uniform tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.GlorotUniformCompromise()
    layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)
    ```
    """
    __name__ = "Glorot Uniform Compromise"

    def __init__(self, seed=None):
        super(GlorotUniformCompromise, self).__init__(distribution="uniform", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(3. / (2. * fan_in)), tf.math.sqrt(3. / (2. * fan_out))]
        r_limit = tf.math.sqrt(6. / (fan_in + fan_out))
        return self.get_random_tensor(shape, c_limit, r_limit, dtype)


class HeNormal(RandomInitializer):
    """
    He normal initializer.
    Reference: https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html
    It draws samples from a truncated normal distribution centered on 0 with
        - Real case: `stddev = sqrt(2 / fan_in)`
        - Complex case: real part stddev = complex part stddev = `1 / sqrt(fan_in)`
    where fan_in is the number of input units in the weight tensor.

    ```
    # Standalone usage:
    import cvnn
    initializer = cvnn.initializers.HeNormal()
    values = initializer(shape=(2, 2))                  # Returns a complex He Normal tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.HeNormal()
    layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)
    ```
    """
    __name__ = "He Normal"

    def __init__(self, seed=None):
        """
        :param seed: A Python integer. An initializer created with a given seed will always
            produce the same random tensor for a given shape and dtype.
        """
        super(HeNormal, self).__init__(distribution="normal", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        """
        Returns a tensor object initialized as specified by the initializer.
        :param shape: Shape of the tensor.
<<<<<<< HEAD
        :param dtype: Optional dtype of the tensor. Either floating or complex. ex: tf.complex64 or tf.float32
=======
        :param dtype: Optional dtype of the tensor. Either floating or complex. ex: tf.complex63 or tf.float32
>>>>>>> 2a18681b0f0ec5e3a1787b8dd7287f1f3f0de985
        """
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(1. / fan_in), tf.math.sqrt(1. / fan_in)]
        r_limit = tf.math.sqrt(2. / fan_in)
        return self.get_random_tensor(shape, c_limit, r_limit, dtype)


class HeUniform(RandomInitializer):
    """
    The He Uniform initializer.
    Reference: http://proceedings.mlr.press/v9/glorot10a.html
    Draws samples from a uniform distribution:
        - Real case: `x ~ U[-limit, limit]` where `limit = sqrt(6 / fan_in)`
        - Complex case: `z / Re{z} = Im{z} ~ U[-limit, limit]` where `limit = sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    ```
    # Standalone usage:
    import cvnn
    initializer = cvnn.initializers.HeUniform()
    values = initializer(shape=(2, 2))                  # Returns a complex He Uniform tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.HeUniform()
    layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)
    ```
    """
    __name__ = "He Uniform"

    def __init__(self, seed=None):
        super(HeUniform, self).__init__(distribution="uniform", seed=seed)

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        fan_in, fan_out = self._compute_fans(shape)
        c_limit = [tf.math.sqrt(3. / fan_in), tf.math.sqrt(3. / fan_in)]
        r_limit = tf.math.sqrt(6. / fan_in)
        return self.get_random_tensor(shape, c_limit, r_limit, dtype)


class Zeros:
    """
    Creates a tensor with all elements set to zero.

    ```
    > >> cvnn.initializers.Zeros()(shape=(2,2))
    <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
    array([[0.+0.j, 0.+0.j],
          [0.+0.j, 0.+0.j]], dtype=complex64)>
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.Zeros()
    layer = cvnn.layers.Dense(input_size=23, output_size=45, bias_initializer=initializer)
    ```
    """
    __name__ = "Zeros"

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.zeros(shape, dtype=dtype)


if __name__ == '__main__':
    # Nothing yet
    set_trace()

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.10'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
