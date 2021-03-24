import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Optional
from cvnn.layers.core import ComplexLayer
from cvnn.layers.core import DEFAULT_COMPLEX_TYPE


class ComplexUpSampling2D(Layer, ComplexLayer):

    def __init__(self, size=(2, 2), data_format: Optional[str] = None, interpolation: str = 'nearest',
                 align_corners: bool = False, dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexUpSampling2D, self).__init__(dtype=self.my_dtype.real_dtype, **kwargs)
        self.align_corners = align_corners
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
        # Equations
        #   https://www.ajdesigner.com/phpinterpolation/linear_interpolation_equation.php
        # Difference with align corners image:
        #   https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        # Examples:
        #   https://www.omnicalculator.com/math/bilinear-interpolation
        #   https://blogs.sas.com/content/iml/2020/05/18/what-is-bilinear-interpolation.html#:~:text=Bilinear%20interpolation%20is%20a%20weighted,the%20point%20and%20the%20corners.&text=The%20only%20important%20formula%20is,x%20%5B0%2C1%5D.
        # Implementations
        #   https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
        if inputs.dtype.is_integer:     # TODO: Check input is a tensor?
            inputs = tf.cast(inputs, dtype=tf.float32)
        i_output = tf.reshape(tf.constant([], dtype=inputs.dtype),
                              (0, desired_size[1], tf.shape(inputs)[2], tf.shape(inputs)[3]))
        j_output = tf.reshape(tf.constant([], dtype=inputs.dtype), (1, 0, tf.shape(inputs)[2], tf.shape(inputs)[3]))
        for x in range(0, desired_size[0]):
            for y in range(0, desired_size[1]):
                (q11, q21, q12, q22), (x1, x2), (y1, y2) = self._get_q_points(x, y, inputs)
                x2_diff = x2 - tf.cast(x, dtype=inputs.dtype)
                x1_diff = tf.cast(x, dtype=inputs.dtype) - x1
                y2_diff = y2 - tf.cast(y, dtype=inputs.dtype)
                y1_diff = tf.cast(y, dtype=inputs.dtype) - y1
                delta_x = x2 - x1
                delta_y = y2 - y1
                if x1 == x2:    # The index was exact, so just make both 1/2
                    x2_diff = tf.cast(0.5, dtype=inputs.dtype)
                    x1_diff = tf.cast(0.5, dtype=inputs.dtype)
                    delta_x = tf.cast(1, dtype=inputs.dtype)
                if y1 == y2:
                    y2_diff = tf.cast(0.5, dtype=inputs.dtype)
                    y1_diff = tf.cast(0.5, dtype=inputs.dtype)
                    delta_y = tf.cast(1, dtype=inputs.dtype)
                t11 = q11 * y2_diff * x2_diff
                t21 = q21 * x1_diff * y2_diff
                t12 = q12 * x2_diff * y1_diff
                t22 = q22 * x1_diff * y1_diff
                to_append = tf.expand_dims(tf.expand_dims((t11 + t22 + t12 + t21) / (delta_y*delta_x), axis=0), axis=0)
                # set_trace()
                j_output = tf.concat([j_output, to_append], axis=1)
            i_output = tf.concat([i_output, j_output], axis=0)
            j_output = tf.reshape(tf.constant([], dtype=inputs.dtype), (1, 0, tf.shape(inputs)[2], tf.shape(inputs)[3]))
        return i_output

    def _get_q_points(self, x, y, inputs):
        X_small = tf.linspace(0, tf.shape(inputs)[0], tf.shape(inputs)[0])
        Y_small = tf.linspace(0, tf.shape(inputs)[1], tf.shape(inputs)[1])
        X, Y = self._to_big((X_small, Y_small))
        x2, x1, y2, y1  = self._get_closest_points(x, y, X, Y)
        q11 = inputs[tf.where(x1 == X)][tf.where(y1 == Y)]
        q21 = inputs[tf.where(x2 == X)][tf.where(y1 == Y)]
        q12 = inputs[tf.where(x1 == X)][tf.where(y2 == Y)]
        q22 = inputs[tf.where(x2 == X)][tf.where(y2 == Y)]
        return (q11, q21, q12, q22), (x1, x2), (y1, y2)


    @staticmethod
    def _get_closest_points(x, y, X, Y):
        pass

    def _to_big(self, index):
        # This must use different equations according to align_corners
        (x_floor * (2 * self.factor_upsample[0]) + 1) / self.factor_upsample[0]
        return index    # Return the big index

    def _get_4_closest_points(self, x, y, x_list, y_list):
        # This function gets the 4 closests points
        X = x2, x1
        Y = y2, y1
        return X, Y

    @staticmethod
    def _get_q_points_align(x, y, inputs, desired_size):
        i_multiplier = (desired_size[0] - 1) / (tf.shape(inputs)[0] - 1)
        j_multiplier = (desired_size[1] - 1) / (tf.shape(inputs)[1] - 1)
        x = x / i_multiplier  # Supposed position of my new element in the small/normalized scale.
        y = y / j_multiplier
        x1 = tf.math.floor(x)
        x2 = tf.math.ceil(x)
        y1 = tf.math.floor(y)
        y2 = tf.math.ceil(y)
        q11 = inputs[tf.cast(x1, tf.int32)][tf.cast(y1, tf.int32)]
        q21 = inputs[tf.cast(x2, tf.int32)][tf.cast(y1, tf.int32)]
        q12 = inputs[tf.cast(x1, tf.int32)][tf.cast(y2, tf.int32)]
        q22 = inputs[tf.cast(x2, tf.int32)][tf.cast(y2, tf.int32)]
        Q = (q11, q21, q12, q22)
        X = (tf.cast(x1 * i_multiplier, dtype=inputs.dtype), tf.cast(x2 * i_multiplier, dtype=inputs.dtype))
        Y = (tf.cast(y1 * i_multiplier, dtype=inputs.dtype), tf.cast(y2 * i_multiplier, dtype=inputs.dtype))
        return Q, X, Y

    def get_real_equivalent(self):
        return ComplexUpSampling2D(size=self.factor_upsample, data_format=self.data_format,
                                   interpolation=self.interpolation, dtype=self.my_dtype.real_dtype)


def test_corners_aligned():
    # Pytorch examples
    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    x = tf.convert_to_tensor([[[[1., 2.], [3., 4.]]]])
    z = tf.complex(real=x, imag=x)
    expected = np.array([[[[1.0000, 1.3333, 1.6667, 2.0000],
                           [1.6667, 2.0000, 2.3333, 2.6667],
                           [2.3333, 2.6667, 3.0000, 3.3333],
                           [3.0000, 3.3333, 3.6667, 4.0000]]]])
    upsample = ComplexUpSampling2D(size=2, interpolation='bilinear', data_format='channels_first', align_corners=True)
    y_complex = upsample(z)
    assert np.allclose(expected, tf.math.real(y_complex).numpy(), 0.0001)
    x = tf.convert_to_tensor([[[[1., 2., 0.],
                                [3., 4., 0.],
                                [0., 0., 0.]]]])
    expected = np.array([[[[1.0000, 1.4000, 1.8000, 1.6000, 0.8000, 0.0000],
                           [1.8000, 2.2000, 2.6000, 2.2400, 1.1200, 0.0000],
                           [2.6000, 3.0000, 3.4000, 2.8800, 1.4400, 0.0000],
                           [2.4000, 2.7200, 3.0400, 2.5600, 1.2800, 0.0000],
                           [1.2000, 1.3600, 1.5200, 1.2800, 0.6400, 0.0000],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
    upsample = ComplexUpSampling2D(size=2, interpolation='bilinear', data_format='channels_first', align_corners=True)
    y = upsample(x)
    assert np.allclose(expected, tf.math.real(y).numpy(), 0.00001)

    # https://blogs.sas.com/content/iml/2020/05/18/what-is-bilinear-interpolation.html#:~:text=Bilinear%20interpolation%20is%20a%20weighted,the%20point%20and%20the%20corners.&text=The%20only%20important%20formula%20is,x%20%5B0%2C1%5D.
    x = tf.convert_to_tensor([[[[0., 4.], [2., 1.]]]])
    z = tf.complex(real=x, imag=x)
    upsample = ComplexUpSampling2D(size=3, interpolation='bilinear', data_format='channels_first', align_corners=True)
    y_complex = upsample(z)
    expected = np.array([[[[0. + 0.j, 0.8 + 0.8j,
                            1.6 + 1.6j, 2.4 + 2.4j,
                            3.2 + 3.2j, 4. + 4.j],
                           [0.4 + 0.4j, 1. + 1.j,
                            1.6 + 1.6j, 2.2 + 2.2j,
                            2.8 + 2.8j, 3.4 + 3.4j],
                           [0.8 + 0.8j, 1.2 + 1.2j,
                            1.6 + 1.6j, 2. + 2.j,
                            2.4 + 2.4j, 2.8 + 2.8j],
                           [1.2 + 1.2j, 1.4 + 1.4j,
                            1.6 + 1.6j, 1.8 + 1.8j,
                            2. + 2.j, 2.2 + 2.2j],
                           [1.6 + 1.6j, 1.6 + 1.6j,
                            1.6 + 1.6j, 1.6 + 1.6j,
                            1.6 + 1.6j, 1.6 + 1.6j],
                           [2. + 2.j, 1.8 + 1.8j,
                            1.6 + 1.6j, 1.4 + 1.4j,
                            1.2 + 1.2j, 1. + 1.j]]]])
    assert np.allclose(expected, y_complex.numpy(), 0.000001)


if __name__ == '__main__':
    from pdb import set_trace
    import numpy as np

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D
    input_shape = (2, 2, 1, 3)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    y_tf = tf.keras.layers.UpSampling2D(size=(1, 2), interpolation='bilinear')(x)
    y_own = ComplexUpSampling2D(size=(1, 2), interpolation='bilinear')(x)
    assert np.all(y_tf == y_own)
    # Pytorch
    #   https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    x = tf.convert_to_tensor([[[[1., 2.], [3., 4.]]]])
    z = tf.complex(real=x, imag=x)
    y_tf = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')(x)
    y_own = ComplexUpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')(z)
    # set_trace()
    assert np.all(y_tf == tf.math.real(y_own).numpy())
    x = tf.convert_to_tensor([[[[1., 2., 0.],
                                [3., 4., 0.],
                                [0., 0., 0.]]]])
    z = tf.complex(real=x, imag=x)
    y_tf = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')(x)
    y_own = ComplexUpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')(z)
    assert np.all(y_tf == tf.math.real(y_own).numpy())
    x = tf.convert_to_tensor([[[[1., 2.], [3., 4.]]]])
    z = tf.complex(real=x, imag=x)
    y_tf = tf.keras.layers.UpSampling2D(size=3, interpolation='bilinear', data_format='channels_first')(x)
    y_own = ComplexUpSampling2D(size=3, interpolation='bilinear', data_format='channels_first')(z)
    set_trace()
    assert np.all(y_tf == tf.math.real(y_own).numpy())
    y_tf = tf.keras.layers.UpSampling2D(size=6, interpolation='bilinear', data_format='channels_first')(x)
    y_own = ComplexUpSampling2D(size=6, interpolation='bilinear', data_format='channels_first')(z)
    assert np.all(y_tf == tf.math.real(y_own).numpy())
    y_tf = tf.keras.layers.UpSampling2D(size=8, interpolation='bilinear', data_format='channels_first')(x)
    y_own = ComplexUpSampling2D(size=8, interpolation='bilinear', data_format='channels_first')(z)
    assert np.all(y_tf == tf.math.real(y_own).numpy())
    test_corners_aligned()
    # to test bicubic= https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/17
