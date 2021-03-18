import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from abc import abstractmethod
# Typing
from typing import Union, Optional, Tuple
# Own models
from cvnn.layers.core import ComplexLayer
from cvnn.layers.core import DEFAULT_COMPLEX_TYPE


class ComplexPooling2D(Layer, ComplexLayer):
    """
    Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
    Abstract class. This class only exists for code reuse. It will never be an exposed API.
    """

    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: str = 'valid', data_format: Optional[str] = None,
                 name: Optional[str] = None, dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        """
        :param pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
            specifying the size of the pooling window.
            Can be a single integer to specify the same value for all spatial dimensions.
        :param strides: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for all spatial dimensions.
        :param padding: A string. The padding method, either 'valid' or 'same'. Case-insensitive.
        :param data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, height, width)`.
        :param name: A string, the name of the layer.
        """
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2,
                                                    'pool_size')  # Values are checked here. No need to check them latter.
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    @abstractmethod
    def pool_function(self, inputs, ksize, strides, padding, data_format):
        pass

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4))
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format
        }
        base_config = super(ComplexPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ComplexMaxPooling2D(ComplexPooling2D):
    """
    Max pooling operation for 2D spatial data.
    Works for complex dtype using the absolute value to get the max.
    """

    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: str = 'valid', data_format: Optional[str] = None,
                 name: Optional[str] = None, **kwargs):
        super(ComplexMaxPooling2D, self).__init__(pool_size=pool_size, strides=strides, padding=padding,
                                                  data_format=data_format, name=name, **kwargs)
        self.argmax = None

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        # The max is calculated with the absolute value. This will still work on real values.
        abs_in = tf.math.abs(inputs)
        output, argmax = tf.nn.max_pool_with_argmax(input=abs_in, ksize=ksize, strides=strides,
                                                    padding=padding, data_format=data_format,
                                                    include_batch_in_index=True)
        self.argmax = argmax
        shape = tf.shape(output)
        tf_res = tf.reshape(tf.gather(tf.reshape(inputs, [-1]), argmax), shape)
        # assert np.all(tf_res == output)             # For debugging when the input is real only!
        assert tf_res.dtype == inputs.dtype
        return tf_res

    def get_real_equivalent(self):
        return ComplexMaxPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")

    def get_max_index(self):
        if self.argmax is None:
            raise AttributeError("Variable argmax did not exist, call at least once the max-pooling layer")
        return self.argmax  # TODO: Shall I check this is use only once?


class ComplexAvgPooling2D(ComplexPooling2D):

    def pool_function(self, inputs, ksize, strides, padding, data_format):
        inputs_r = tf.math.real(inputs)
        inputs_i = tf.math.imag(inputs)
        output_r = tf.nn.avg_pool2d(input=inputs_r, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        output_i = tf.nn.avg_pool2d(input=inputs_i, ksize=ksize, strides=strides,
                                    padding=padding, data_format=data_format)
        if inputs.dtype.is_complex:
            output = tf.complex(output_r, output_i)
        else:
            output = output_r
        return output

    def get_real_equivalent(self):
        return ComplexAvgPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding,
                                   data_format=self.data_format, name=self.name + "_real_equiv")


class ComplexUnPooling2D(Layer, ComplexLayer):

    def __init__(self, trainable=True, name=None, dtype=DEFAULT_COMPLEX_TYPE, dynamic=False, **kwargs):
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexUnPooling2D, self).__init__(trainable=trainable, name=name, dtype=self.my_dtype.real_dtype,
                                               dynamic=dynamic, **kwargs)

    def call(self, inputs, output_shape, unpool_mat=None, data_format='channels_last', **kwargs):
        # https://stackoverflow.com/a/42549265/5931672
        # TODO:
        #   1. Verify output_shape first element.
        #   2. Verify None shape tensor for inputs.
        flatten_inputs = tf.reshape(inputs, [-1])
        output_flatten_size = tf.cast(tf.math.reduce_prod(output_shape), tf.int64)
        argmax_flatten = tf.reshape(unpool_mat, [-1])
        indices = tf.convert_to_tensor([[0, a] for a in argmax_flatten])
        assert flatten_inputs.shape == argmax_flatten.shape, f"Flatten input shape {flatten_inputs.shape} was not " \
                                                             f"equal to flatten argmax shape {argmax_flatten.shape}"
        # set_trace()
        output_flatten = tf.sparse.SparseTensor(indices=indices, values=flatten_inputs, dense_shape=(1, output_flatten_size.numpy()))
        output_flatten = tf.sparse.to_dense(output_flatten)
        set_trace()
        # for value, index in zip(flatten_inputs, argmax_flatten):
        #     output_flatten = output_flatten[index].assign(value)
        output = tf.reshape(output_flatten, shape=output_shape)
        return output

    def get_real_equivalent(self):
        return ComplexUnPooling2D(trainable=self.trainable, name=self.name, dtype=self.my_dtype.real_dtype,
                                  dynamic=self.dtype)


if __name__ == "__main__":
    import numpy as np
    from pdb import set_trace

    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 7, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ], [
        [0, 4, 5],
        [3, 2, 2],
        [4, 8, 9]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i
    img = np.reshape(img, (2, 3, 3, 1))
    max_pool = ComplexMaxPooling2D(strides=1, data_format="channels_last")
    res = max_pool(img.astype(np.complex64))
    print(img.reshape(2, 3, 3))
    print(res.numpy().reshape(2, 2, 2))
    print(max_pool.get_max_index().numpy().reshape(2, 2, 2))
    # max_pool.input_shape -> *** AttributeError: The layer has never been called and thus has no defined input shape.
    max_unpooling = ComplexUnPooling2D()
    unpooled = max_unpooling(res, unpool_mat=max_pool.get_max_index(), output_shape=img.shape)
    set_trace()
