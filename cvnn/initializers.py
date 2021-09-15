from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.keras.initializers import Initializer
import sys
from pdb import set_trace
# Typing
from typing import Optional

INIT_TECHNIQUES = {'zero_imag', 'mirror'}


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


class ComplexInitializer(Initializer):

    def __init__(self, distribution: str = "uniform", seed: Optional[int] = None):
        if distribution.lower() not in {"uniform", "normal"}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        else:
            self.distribution = distribution.lower()
        self._random_generator = _RandomGenerator(seed)

    def _call_random_generator(self, shape, arg, dtype):
        if self.distribution == "uniform":
            return self._random_generator.random_uniform(shape=shape, minval=-arg, maxval=arg, dtype=dtype)
        elif self.distribution == "normal":
            # I make this magic number division because that's what tf does on this case
            return self._random_generator.truncated_normal(shape=shape, mean=0.0, stddev=arg / .87962566103423978,
                                                           dtype=dtype)

    @abstractmethod
    def _compute_limit(self, fan_in, fan_out):
        pass

    def __call__(self, shape, dtype=tf.dtypes.complex64, **kwargs):
        fan_in, fan_out = _compute_fans(shape)
        arg = self._compute_limit(fan_in, fan_out)
        dtype = tf.dtypes.as_dtype(dtype)
        if dtype.is_complex:
            arg = arg / np.sqrt(2)
        return self._call_random_generator(shape=shape, arg=arg, dtype=dtype.real_dtype)

    def get_config(self):  # To support serialization
        return {"seed": self._random_generator.seed}


class ComplexGlorotUniform(ComplexInitializer):
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
    initializer = cvnn.initializers.ComplexGlorotUniform()
    values = initializer(shape=(2, 2))                  # Returns a complex Glorot Uniform tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.ComplexGlorotUniform()
    layer = cvnn.layers.ComplexDense(units=10, kernel_initializer=initializer)
    ```
    """
    __name__ = "Complex Glorot Uniform"

    def __init__(self, seed: Optional[int] = None):
        super(ComplexGlorotUniform, self).__init__(distribution="uniform", seed=seed)

    def _compute_limit(self, fan_in, fan_out):
        return tf.math.sqrt(6. / (fan_in + fan_out))


class ComplexGlorotNormal(ComplexInitializer):
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
    initializer = cvnn.initializers.ComplexGlorotNormal()
    values = initializer(shape=(2, 2))                  # Returns a complex Glorot Normal tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.ComplexGlorotNormal()
    layer = cvnn.layers.ComplexDense(units=10, kernel_initializer=initializer)
    ```
    """
    __name__ = "Complex Glorot Normal"

    def __init__(self, seed: Optional[int] = None):
        super(ComplexGlorotNormal, self).__init__(distribution="normal", seed=seed)

    def _compute_limit(self, fan_in, fan_out):
        return tf.math.sqrt(2. / (fan_in + fan_out))


class ComplexHeUniform(ComplexInitializer):
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
    initializer = cvnn.initializers.ComplexHeUniform()
    values = initializer(shape=(2, 2))                  # Returns a real He Uniform tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.ComplexHeUniform()
    layer = cvnn.layers.ComplexDense(units=10, kernel_initializer=initializer)
    ```
    """
    __name__ = "Complex He Uniform"

    def __init__(self, seed: Optional[int] = None):
        super(ComplexHeUniform, self).__init__(distribution="uniform", seed=seed)

    def _compute_limit(self, fan_in, fan_out):
        return tf.math.sqrt(6. / fan_in)


class ComplexHeNormal(ComplexInitializer):
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
    initializer = cvnn.initializers.ComplexHeNormal()
    values = initializer(shape=(2, 2))                  # Returns a complex He Normal tensor of shape (2, 2)
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.ComplexHeNormal()
    layer = cvnn.layers.ComplexDense(units=10, kernel_initializer=initializer)
    ```
    """
    __name__ = "Complex He Normal"

    def __init__(self, seed: Optional[int] = None):
        super(ComplexHeNormal, self).__init__(distribution="normal", seed=seed)

    def _compute_limit(self, fan_in, fan_out):
        return tf.math.sqrt(2. / fan_in)


class Zeros:
    """
    Creates a tensor with all elements set to zero.

    ```
    > >> cvnn.initializers.Zeros()(shape=(2,2))
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[0.+0.j, 0.+0.j],
          [0.+0.j, 0.+0.j]], dtype=float32)>
    ```

    ```
    # Usage in a cvnn layer:
    import cvnn
    initializer = cvnn.initializers.Zeros()
    layer = cvnn.layers.ComplexDense(units=10, bias_initializer=initializer)
    ```
    """
    __name__ = "Zeros"

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.zeros(shape, dtype=tf.dtypes.as_dtype(dtype).real_dtype)


class Ones:
    __name__ = "Ones"

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.ones(shape, dtype=tf.dtypes.as_dtype(dtype).real_dtype)


init_dispatcher = {
    "ComplexGlorotUniform": ComplexGlorotUniform,
    "ComplexGlorotNormal": ComplexGlorotNormal,
    "ComplexHeUniform": ComplexHeUniform,
    "ComplexHeNormal": ComplexHeNormal
}


if __name__ == '__main__':
    # Nothing yet
    set_trace()

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.13'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
