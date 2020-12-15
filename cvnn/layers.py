import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, InputLayer
from tensorflow import TensorShape, Tensor
import numpy as np
from cvnn import logger
from pdb import set_trace
# Typing
from typing import Union, List

t_input = Union[Tensor, tuple, list]
t_input_shape = Union[TensorShape, List[TensorShape]]

DEFAULT_COMPLEX_TYPE = np.complex64


def iscomplex(inputs: t_input):
    return inputs.dtype.is_complex


class ComplexInput(InputLayer):

    def __init__(self, input_shape=None, batch_size=None, dtype=DEFAULT_COMPLEX_TYPE, input_tensor=None, sparse=False,
                 name=None, ragged=False, **kwargs):
        super(ComplexInput, self).__init__(input_shape=input_shape, batch_size=batch_size, dtype=dtype,
                                           input_tensor=input_tensor, sparse=sparse,
                                           name=name, ragged=ragged, **kwargs
                                           )

   
class ComplexFlatten(Flatten):
    
    def call(self, inputs: t_input):
        # tf.print(f"inputs at ComplexFlatten are {inputs.dtype}")
        real_flat = super(ComplexFlatten, self).call(tf.math.real(inputs))
        imag_flat = super(ComplexFlatten, self).call(tf.math.imag(inputs))
        return tf.cast(tf.complex(real_flat, imag_flat), inputs.dtype)      # Keep input dtype


class ComplexDense(Dense):
    
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dtype=DEFAULT_COMPLEX_TYPE,
                 **kwargs):
        super(ComplexDense, self).__init__(units, activation=activation, use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer, **kwargs)
        # Cannot override dtype of the layer because it has a read-only @property
        self.my_dtype = tf.dtypes.as_dtype(dtype)
         
    def build(self, input_shape):
        if self.my_dtype.is_complex:
            self.w_r = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer=self.kernel_initializer,
                trainable=True,
            )
            self.w_i = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer=self.kernel_initializer,
                trainable=True,
            )
            self.b_r = self.add_weight(
                shape=(self.units,), initializer=self.bias_initializer, trainable=True
            )
            self.b_i = self.add_weight(
                shape=(self.units,), initializer=self.bias_initializer, trainable=True
            )
        else:
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                dtype=self.my_dtype,
                initializer=self.kernel_initializer,
                trainable=True,
            )
            self.b = self.add_weight(shape=(self.units,), dtype=self.my_dtype,
                                     initializer=self.bias_initializer, trainable=True)

    def call(self, inputs: t_input):
        # tf.print(f"inputs at ComplexDense are {inputs.dtype}")
        if inputs.dtype != self.my_dtype:
            tf.print(f"Expected input to be {self.my_dtype}, but received {inputs.dtype}.\n"
                     f"You might have forgotten to use tf.keras.Input(shape, dtype=np.complex128).")
            # logger.warning(f"Input expected to be {self.my_dtype}, but received {inputs.dtype}.") 
            inputs = tf.cast(inputs, self.my_dtype)
        if self.my_dtype.is_complex:
            w = tf.cast(tf.complex(self.w_r, self.w_i), self.my_dtype)
            b = tf.cast(tf.complex(self.b_r, self.b_i), self.my_dtype)
        else:
            w = self.w
            b = self.b
        out = tf.matmul(inputs, w) + b
        return self.activation(out)


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.30'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
