import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Optional
from cvnn.layers.core import ComplexLayer
from cvnn.layers.core import DEFAULT_COMPLEX_TYPE


class ComplexUpSampling2D(Layer, ComplexLayer):

    def __init__(self, size=(2, 2), data_format: Optional[str] = None, interpolation: str = 'nearest',
                 dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexUpSampling2D, self).__init__(dtype=self.my_dtype.real_dtype, **kwargs)

        if isinstance(size, int):
            self.factor_upsample = (size,) * 2
        else:
            self.factor_upsample = tuple(size)  # Python will tell me if this is not possible
        self.interpolation = interpolation.lower()
        if self.interpolation not in {'nearest', 'bilinear'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` or `"bilinear"`.')
        if data_format is None:
            data_format = 'channels_last'
        self.data_format = data_format.lower()
        if self.data_format not in {'channels_first', 'channels_last'}:
            raise ValueError(f'The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            inputs = tf.transpose(inputs, perm=[1, 2, 0, 3])
        elif self.data_format == 'channels_first':  # I checked it at init, shall I check again?
            inputs = tf.transpose(inputs, perm=[2, 3, 0, 1])
        else:
            raise ValueError(f'The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')
        desired_output_shape = [i * o for i, o in zip(inputs.shape, self.factor_upsample)]
        assert len(desired_output_shape) == 2  # The for will do only for the shortest so I should be Ok.
        if self.interpolation == 'nearest':
            # output = tf.repeat(input=tf.repeat(input=inputs, repeats=(self.factor_upsample[0],)*inputs.shape[0],
            #                                    axis=0),
            #                    repeats=(self.factor_upsample[1],)*inputs.shape[1], axis=1)
            output = self.nearest_neighbor(inputs, desired_output_shape)
        elif self.interpolation == 'bilinear':
            output = self.bilinear(inputs, desired_output_shape)
        else:
            raise ValueError(f"Unknown interpolation {self.interpolation}")
        if self.data_format == 'channels_last':
            output = tf.transpose(output, perm=[2, 0, 1, 3])
        elif self.data_format == 'channels_first':  # I checked it at init, shall I check again?
            output = tf.transpose(output, perm=[2, 3, 0, 1])
        else:
            raise ValueError(f'The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')
        return output

    @staticmethod
    def nearest_neighbor(inputs_to_resize, desired_size):
        i_output = tf.reshape(tf.constant([], dtype=inputs_to_resize.dtype),
                              (0, desired_size[1], tf.shape(inputs_to_resize)[2], tf.shape(inputs_to_resize)[3]))
        j_output = tf.reshape(tf.constant([], dtype=inputs_to_resize.dtype),
                              (1, 0, tf.shape(inputs_to_resize)[2], tf.shape(inputs_to_resize)[3]))
        for i in range(0, desired_size[0]):
            for j in range(0, desired_size[1]):
                i_new = tf.cast(tf.floor((inputs_to_resize.shape[0] * i) / desired_size[0]), dtype=tf.int32)
                j_new = tf.cast(tf.floor((inputs_to_resize.shape[1] * j) / desired_size[1]), dtype=tf.int32)
                to_append = tf.expand_dims(tf.expand_dims(inputs_to_resize[i_new, j_new], axis=0), axis=0)
                j_output = tf.concat([j_output, to_append], axis=1)
            i_output = tf.concat([i_output, j_output], axis=0)
            j_output = tf.reshape(tf.constant([], dtype=inputs_to_resize.dtype),
                                  (1, 0, tf.shape(inputs_to_resize)[2], tf.shape(inputs_to_resize)[3]))
        # output = tf.transpose(i_output, perm=[2, 0, 1, 3])
        # assert i_output.shape == (tuple(deisred_size) + (input.shape[2], input.shape[3]))
        return i_output

    def bilinear(self, inputs, desired_size):
        i_output = tf.reshape(tf.constant([], dtype=inputs.dtype),
                              (0, desired_size[1], tf.shape(inputs)[2], tf.shape(inputs)[3]))
        j_output = tf.reshape(tf.constant([], dtype=inputs.dtype), (1, 0, tf.shape(inputs)[2], tf.shape(inputs)[3]))
        i_multiplier = desired_size[0] - 1
        j_multiplier = desired_size[1] - 1
        for i in range(0, desired_size[0]):
            for j in range(0, desired_size[1]):
                x = i / i_multiplier
                y = j / j_multiplier

                x1 = tf.math.floor(x)
                x2 = tf.math.ceil(x)
                y1 = tf.math.floor(y)
                y2 = tf.math.ceil(y)

                x2_diff = tf.cast(x2 - x, dtype=inputs.dtype)
                x1_diff = tf.cast(x - x1, dtype=inputs.dtype)
                y2_diff = tf.cast(y2 - y, dtype=inputs.dtype)
                y1_diff = tf.cast(y - y1, dtype=inputs.dtype)
                if x1 == x2:
                    x2_diff = tf.cast(0.5, dtype=inputs.dtype)
                    x1_diff = tf.cast(0.5, dtype=inputs.dtype)
                if y1 == y2:
                    y2_diff = tf.cast(0.5, dtype=inputs.dtype)
                    y1_diff = tf.cast(0.5, dtype=inputs.dtype)

                q11 = inputs[tf.cast(x1, tf.int32)][tf.cast(y1, tf.int32)]
                q21 = inputs[tf.cast(x2, tf.int32)][tf.cast(y1, tf.int32)]
                q12 = inputs[tf.cast(x1, tf.int32)][tf.cast(y2, tf.int32)]
                q22 = inputs[tf.cast(x2, tf.int32)][tf.cast(y2, tf.int32)]
                t11 = q11 * y2_diff * x2_diff
                t21 = q21 * x1_diff * y2_diff
                t12 = q12 * x2_diff * y1_diff
                t22 = q22 * x1_diff * y1_diff
                to_append = tf.expand_dims(tf.expand_dims(t11 + t22 + t12 + t21, axis=0), axis=0)
                # set_trace()
                j_output = tf.concat([j_output, to_append], axis=1)
            i_output = tf.concat([i_output, j_output], axis=0)
            j_output = tf.reshape(tf.constant([], dtype=inputs.dtype), (1, 0, tf.shape(inputs)[2], tf.shape(inputs)[3]))
        return i_output

    def get_real_equivalent(self):
        return ComplexUpSampling2D(size=self.factor_upsample, data_format=self.data_format,
                                   interpolation=self.interpolation, dtype=self.my_dtype.real_dtype)


if __name__ == '__main__':
    from pdb import set_trace
    import numpy as np

    x = tf.convert_to_tensor([[[[1., 2.], [3., 4.]]]])
    z = tf.complex(real=x, imag=x)
    upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')
    y_tf = upsample(x)
    upsample = ComplexUpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')
    y_complex = upsample(z)

    x = tf.convert_to_tensor([[[[0., 4.], [2., 1.]]]])
    z = tf.complex(real=x, imag=x)
    upsample = ComplexUpSampling2D(size=3, interpolation='bilinear', data_format='channels_first')
    y_complex = upsample(z)
    set_trace()
