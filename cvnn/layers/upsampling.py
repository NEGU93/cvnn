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
            # TODO: Could we make this with tf.repeat?
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
                              (0, desired_size[1], inputs_to_resize.shape[2], inputs_to_resize.shape[3]))
        j_output = tf.reshape(tf.constant([], dtype=inputs_to_resize.dtype),
                              (1, 0, inputs_to_resize.shape[2], inputs_to_resize.shape[3]))
        for i in range(0, desired_size[0]):
            for j in range(0, desired_size[1]):
                i_new = tf.cast(tf.floor((inputs_to_resize.shape[0] * i) / desired_size[0]), dtype=tf.int32)
                j_new = tf.cast(tf.floor((inputs_to_resize.shape[1] * j) / desired_size[1]), dtype=tf.int32)
                to_apend = tf.expand_dims(tf.expand_dims(inputs_to_resize[i_new, j_new], axis=0), axis=0)
                j_output = tf.concat([j_output, to_apend], axis=1)
            i_output = tf.concat([i_output, j_output], axis=0)
            j_output = tf.reshape(tf.constant([], dtype=inputs_to_resize.dtype),
                                  (1, 0, inputs_to_resize.shape[2], inputs_to_resize.shape[3]))
        # output = tf.transpose(i_output, perm=[2, 0, 1, 3])
        # assert i_output.shape == (tuple(deisred_size) + (input.shape[2], input.shape[3]))
        return i_output

    @staticmethod
    def bilinear(input, desired_size):
        pass

    def get_real_equivalent(self):
        return ComplexUpSampling2D(size=self.factor_upsample, data_format=self.data_format,
                                   interpolation=self.interpolation, dtype=self.my_dtype.real_dtype)


if __name__ == '__main__':
    import numpy as np
    from pdb import set_trace

    input_shape = (2, 2, 1, 3)
    x = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    z = tf.complex(real=x, imag=x)
    upsample = ComplexUpSampling2D(size=(2, 3))
    y = upsample(z)
    expected = np.array([[[[0. + 0.j, 1. + 1.j, 2. + 2.j],
                           [0. + 0.j, 1. + 1.j, 2. + 2.j],
                           [0. + 0.j, 1. + 1.j, 2. + 2.j]],
                          [[0. + 0.j, 1. + 1.j, 2. + 2.j],
                           [0. + 0.j, 1. + 1.j, 2. + 2.j],
                           [0. + 0.j, 1. + 1.j, 2. + 2.j]],
                          [[3. + 3.j, 4. + 4.j, 5. + 5.j],
                           [3. + 3.j, 4. + 4.j, 5. + 5.j],
                           [3. + 3.j, 4. + 4.j, 5. + 5.j]],
                          [[3. + 3.j, 4. + 4.j, 5. + 5.j],
                           [3. + 3.j, 4. + 4.j, 5. + 5.j],
                           [3. + 3.j, 4. + 4.j, 5. + 5.j]]],
                         [[[6. + 6.j, 7. + 7.j, 8. + 8.j],
                           [6. + 6.j, 7. + 7.j, 8. + 8.j],
                           [6. + 6.j, 7. + 7.j, 8. + 8.j]],
                          [[6. + 6.j, 7. + 7.j, 8. + 8.j],
                           [6. + 6.j, 7. + 7.j, 8. + 8.j],
                           [6. + 6.j, 7. + 7.j, 8. + 8.j]],
                          [[9. + 9.j, 10. + 10.j, 11. + 11.j],
                           [9. + 9.j, 10. + 10.j, 11. + 11.j],
                           [9. + 9.j, 10. + 10.j, 11. + 11.j]],
                          [[9. + 9.j, 10. + 10.j, 11. + 11.j],
                           [9. + 9.j, 10. + 10.j, 11. + 11.j],
                           [9. + 9.j, 10. + 10.j, 11. + 11.j]]]])
    assert np.all(y.numpy() == expected)
    upsample = ComplexUpSampling2D(size=(1, 3))
    y = upsample(z)
    expected = np.array([[[[0. + 0.j, 1. + 1.j, 2. + 2.j],
                           [0. + 0.j, 1. + 1.j, 2. + 2.j],
                           [0. + 0.j, 1. + 1.j, 2. + 2.j]],
                          [[3. + 3.j, 4. + 4.j, 5. + 5.j],
                           [3. + 3.j, 4. + 4.j, 5. + 5.j],
                           [3. + 3.j, 4. + 4.j, 5. + 5.j]]],
                         [[[6. + 6.j, 7. + 7.j, 8. + 8.j],
                           [6. + 6.j, 7. + 7.j, 8. + 8.j],
                           [6. + 6.j, 7. + 7.j, 8. + 8.j]],
                          [[9. + 9.j, 10. + 10.j, 11. + 11.j],
                           [9. + 9.j, 10. + 10.j, 11. + 11.j],
                           [9. + 9.j, 10. + 10.j, 11. + 11.j]]]])
    assert np.all(y.numpy() == expected)
    upsample = ComplexUpSampling2D(size=(1, 2))
    y = upsample(z)
    # print(y)
    y_tf = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
    my_y = upsample.get_real_equivalent()(x)
    assert np.all(my_y == y_tf)
    x = tf.convert_to_tensor([[[[1., 2.], [3., 4.]]]])
    upsample = ComplexUpSampling2D(size=2, data_format='channels_first')
    my_y = upsample(x)
    y_tf = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_first')(x)
    assert np.all(my_y == y_tf)
