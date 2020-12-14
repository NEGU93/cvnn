import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Flatten, Dense, Layer
from tensorflow import TensorShape, Tensor
import numpy as np
from cvnn import logger
from pdb import set_trace
# Typing
from tensorflow import dtypes
from numpy import dtype, ndarray
from typing import Union, List

t_input = Union[Tensor, tuple, list]
t_input_shape = Union[TensorShape, List[TensorShape]]

def iscomplex(inputs:t_input):
    return inputs.dtype.is_complex
    
   
class ComplexFlatten(Flatten):
    
    def call(self, inputs: t_input):
        real_flat = super(ComplexFlatten, self).call(tf.math.real(inputs))
        imag_flat = super(ComplexFlatten, self).call(tf.math.imag(inputs))
        return tf.cast(tf.complex(real_flat, imag_flat), inputs.dtype)      # Keep input dtype
    
class ComplexDense(Dense):
    
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dtype = tf.complex128,
                 **kwargs):
        super(ComplexDense, self).__init__(units, activation=activation, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer, **kwargs)
        self.my_dtype = tf.dtypes.as_dtype(dtype)    # Cannot override dtype of the layer because it has a read-only @property
        
    
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
                initializer=self.kernel_initializer,
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer=self.bias_initializer, trainable=True
            )

    def call(self, inputs: t_input):
        if inputs.dtype != self.my_dtype:
            tf.print(f"Expected input to be {self.my_dtype}, but received {inputs.dtype}.")
            # logger.warning(f"Input expected to be {self.my_dtype}, but received {inputs.dtype}.") 
            inputs = tf.cast(inputs, self.my_dtype)
        if self.my_dtype.is_complex:
            w = tf.cast(tf.complex(self.w_r, self.w_i), self.my_dtype)
            b = tf.cast(tf.complex(self.b_r, self.b_i), self.my_dtype)
        else:
            w = tf.cast(self.w, self.my_dtype)
            b = tf.cast(self.b, self.my_dtype)
        return tf.matmul(inputs, w) + b


def small_example():
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
    img =img_r + 1j * img_i
    c_flat = ComplexFlatten()
    c_dense = ComplexDense(units=10)
    res = c_dense(c_flat(img.astype(np.complex64)))
    
def serial_layers():
    model = tf.keras.models.Sequential()
    model.add(ComplexDense(32, activation='relu', input_shape=(32, 32, 3)))
    model.add(ComplexDense(32))
    print(model.output_shape)
    
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def mnist_example():
    tf.enable_v2_behavior()
    
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    
    model = tf.keras.models.Sequential([
        ComplexFlatten(input_shape=(28, 28, 1)),
        ComplexDense(128,activation='relu', dtype=tf.float32),
        ComplexDense(10, dtype=tf.float32)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

if __name__ == "__main__":
    dtype = np.float32
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype(dtype)
    test_images = test_images.astype(dtype)
    train_labels = train_labels.astype(dtype)
    test_labels = test_labels.astype(dtype)
    
    model = tf.keras.Sequential([
        ComplexFlatten(input_shape=(28, 28)),
        ComplexDense(128, activation='relu', dtype=np.complex64),
        ComplexDense(10, dtype=np.float32)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )
    model.fit(tf.complex(train_images, train_images), train_labels, epochs=1)
    
    # import pdb; pdb.set_trace()


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.28'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
