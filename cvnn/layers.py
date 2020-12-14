import tensorflow as tf
from tensorflow.keras.layers import Flatten
import numpy as np
from cvnn import logger
from pdb import set_trace
# Typing
from tensorflow import dtypes
from numpy import dtype, ndarray
from typing import Union, Callable, Optional, List, Set


SUPPORTED_DTYPES = (np.complex64, np.float32)  # , np.complex128, np.float64) Gradients return None when complex128

t_input_shape = Union[int, tuple, list]
t_Callable_shape = Union[t_input_shape, Callable]   # Either a input_shape or a function that sets self.output
t_Dtype = Union[dtypes.DType, dtype]

   
class ComplexFlatten(Flatten):
    
    def call(self, inputs):
        real_flat = super(ComplexFlatten, self).call(tf.math.real(inputs))
        imag_flat = super(ComplexFlatten, self).call(tf.math.imag(inputs))
        return tf.complex(real_flat, imag_flat)


if __name__ == "__main__":
    img_r = np.array([[
        [0, 1, 2], 
        [0, 2, 2], 
        [0, 5, 7]
    ],[
        [0, 4, 5], 
        [3, 7, 9], 
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5], 
        [3, 7, 9], 
        [4, 5, 3]
    ],[
        [0, 4, 5], 
        [3, 7, 9], 
        [4, 5, 3]
    ]]).astype(np.float32)
    img = tf.complex(img_r, img_i)
    
    c_flat = ComplexFlatten()
    
    res = c_flat.call(img)
    
    import pdb; pdb.set_trace()


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.28'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
