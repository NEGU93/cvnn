import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Optional, Union, Tuple
from cvnn.layers.core import ComplexLayer
from cvnn.layers.core import DEFAULT_COMPLEX_TYPE


class ComplexUpSampling2D(Layer, ComplexLayer):

    def __init__(self, size: Union[int, Tuple[int, int]] = (2, 2),
                 data_format: Optional[str] = None, interpolation: str = 'nearest',
                 align_corners: bool = False, dtype=DEFAULT_COMPLEX_TYPE, **kwargs):
        """
        :param size: Int, or tuple of 2 integers. The upsampling factors for rows and columns.
        :param data_format: string, one of channels_last (default) or channels_first.
            The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape
            (batch_size, height, width, channels) while channels_first corresponds to inputs with shape
            (batch_size, channels, height, width).
        :param interpolation: A string, one of nearest or bilinear.
        :param align_corners:  if True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at those pixels.
            Example of align coreners: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        """
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexUpSampling2D, self).__init__(dtype=self.my_dtype.real_dtype, **kwargs)
        self.align_corners = align_corners
        if isinstance(size, int):
            self.factor_upsample = (size,) * 2
        else:
            self.factor_upsample = tuple(size)  # Python will tell me if this is not possible
        # TODO: Check is tuple of ints and no negative values!
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
        output = self.upsample(inputs=inputs)
        if self.data_format == 'channels_last':
            output = tf.transpose(output, perm=[2, 0, 1, 3])
        elif self.data_format == 'channels_first':  # I checked it at init, shall I check again?
            output = tf.transpose(output, perm=[2, 3, 0, 1])
        else:
            raise ValueError(f'The `data_format` argument must be one of "channels_first", "channels_last". '
                             f'Received: {self.data_format}')
        return output

    def upsample(self, inputs):
        if inputs.dtype.is_integer:  # TODO: Check input is a tensor?
            inputs = tf.cast(inputs, dtype=tf.float32)
        desired_size = [i * o for i, o in zip(inputs.shape, self.factor_upsample)]
        assert len(desired_size) == 2  # The for will do only for the shortest so I should be Ok.
        i_output = tf.reshape(tf.constant([], dtype=inputs.dtype),
                              (0, desired_size[1], tf.shape(inputs)[2], tf.shape(inputs)[3]))
        j_output = tf.reshape(tf.constant([], dtype=inputs.dtype), (1, 0, tf.shape(inputs)[2], tf.shape(inputs)[3]))
        for x in range(0, desired_size[0]):
            for y in range(0, desired_size[1]):
                if self.interpolation == 'bilinear':
                    to_append = self.bilinear(inputs=inputs, x=x, y=y)
                elif self.interpolation == 'nearest':
                    to_append = self.nearest_neighbor(inputs=inputs, x=x, y=y)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interpolation}")
                to_append = tf.expand_dims(tf.expand_dims(to_append, axis=0), axis=0)
                j_output = tf.concat([j_output, to_append], axis=1)
            i_output = tf.concat([i_output, j_output], axis=0)
            j_output = tf.reshape(tf.constant([], dtype=inputs.dtype), (1, 0, tf.shape(inputs)[2], tf.shape(inputs)[3]))
        return i_output

    def nearest_neighbor(self, inputs, x, y):
        # output = tf.repeat(input=tf.repeat(input=inputs, repeats=(self.factor_upsample[0],)*inputs.shape[0],
        #                                    axis=0),
        #                    repeats=(self.factor_upsample[1],)*inputs.shape[1], axis=1)
        # i_new = tf.cast(tf.floor((inputs.shape[0] * x) / desired_output_shape[0]), dtype=tf.int32)
        # j_new = tf.cast(tf.floor((inputs.shape[1] * y) / desired_output_shape[1]), dtype=tf.int32)
        i_new, j_new = self._get_nearest_neighbour(x, y, inputs)
        to_append = inputs[i_new, j_new]
        # assert i_output.shape == (tuple(deisred_size) + (input.shape[2], input.shape[3]))
        return to_append

    def _get_nearest_neighbour(self, x, y, inputs):
        X_small = tf.linspace(0, tf.shape(inputs)[0] - 1, tf.shape(inputs)[0])
        Y_small = tf.linspace(0, tf.shape(inputs)[1] - 1, tf.shape(inputs)[1])
        X, Y = self._to_big(X_small, Y_small)
        i, j = self._get_closest_point(x, y, X, Y)
        return i, j

    @staticmethod
    def _get_closest_point(x, y, x_list, y_list):
        x_distance = tf.math.square(x_list - x)
        y_distance = tf.math.square(y_list - y)
        x_closest = tf.argmin(x_distance)
        y_closest = tf.argmin(y_distance)
        return x_closest, y_closest

    def bilinear(self, inputs, x, y):
        # Equations
        #   https://www.ajdesigner.com/phpinterpolation/linear_interpolation_equation.php
        # Difference with align corners image:
        #   https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        # Examples:
        #   https://www.omnicalculator.com/math/bilinear-interpolation
        #   https://blogs.sas.com/content/iml/2020/05/18/what-is-bilinear-interpolation.html#:~:text=Bilinear%20interpolation%20is%20a%20weighted,the%20point%20and%20the%20corners.&text=The%20only%20important%20formula%20is,x%20%5B0%2C1%5D.
        # Implementations
        #   https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
        (q11, q21, q12, q22), (x1, x2), (y1, y2) = self._get_q_points(x, y, inputs)
        # There are 3 cases:
        #   1. All 4 q's are different and surround the point
        #   2. There are basically 2 q's (get repeated)
        #   3. All 4 q's are equal
        x2_diff = tf.cast(x2, dtype=inputs.dtype) - tf.cast(x, dtype=inputs.dtype)
        x1_diff = tf.cast(x, dtype=inputs.dtype) - tf.cast(x1, dtype=inputs.dtype)
        y2_diff = tf.cast(y2, dtype=inputs.dtype) - tf.cast(y, dtype=inputs.dtype)
        y1_diff = tf.cast(y, dtype=inputs.dtype) - tf.cast(y1, dtype=inputs.dtype)
        delta_x = tf.cast(x2 - x1, dtype=inputs.dtype)
        delta_y = tf.cast(y2 - y1, dtype=inputs.dtype)
        # The next conditions happens in cases 2 or 3, only one for case 2 and both for case 3.
        # Using the following equations/conditions, the general equation stands for all 3 cases.
        if x1 == x2:  # The index was exact, so just make both 1/2
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
        to_append = (t11 + t22 + t12 + t21) / (delta_y * delta_x)
        return to_append

    def _get_q_points(self, x, y, inputs):
        # 1. Get x and y coordinates of inputs (basically from 0 to the end)
        X_small = tf.linspace(0, tf.shape(inputs)[0] - 1, tf.shape(inputs)[0])
        Y_small = tf.linspace(0, tf.shape(inputs)[1] - 1, tf.shape(inputs)[1])
        # 2. Transform those coordinates into a index equivalent on the new bigger image.
        X, Y = self._to_big(X_small, Y_small)
        # 3. Get the closest points of X to x. x2 is the closest but bigger and x1 is closest but small
        # for example X = [0.5, 2.5 4.5] then
        #   If x is 1, then x1 is 0.5 and x2 is 2.5.
        #   If x is 2.5, both x1 and x2 are 2.5.
        #   If x is 0, then x1 equals x2.
        #   If x is 5 then x2 equals x1
        x2, x1, y2, y1 = self._get_4_closest_points(x, y, X, Y)
        # Get the points according to the coordinates obtained.
        q11 = inputs[tf.where(x1 == X)[0][0]][tf.where(y1 == Y)[0][0]]
        q21 = inputs[tf.where(x2 == X)[0][0]][tf.where(y1 == Y)[0][0]]
        q12 = inputs[tf.where(x1 == X)[0][0]][tf.where(y2 == Y)[0][0]]
        q22 = inputs[tf.where(x2 == X)[0][0]][tf.where(y2 == Y)[0][0]]
        return (q11, q21, q12, q22), (x1, x2), (y1, y2)

    def _to_big(self, x_index, y_index):
        # TODO: Check index dtype
        # This must use different equations according to align_corners
        # set_trace()
        if self.align_corners:
            x_index = tf.linspace(0, len(x_index) * self.factor_upsample[0] - 1, len(x_index))
            y_index = tf.linspace(0, len(y_index) * self.factor_upsample[1] - 1, len(y_index))
        else:
            x_index = (x_index + 0.5) * self.factor_upsample[0] - 0.5
            y_index = (y_index + 0.5) * self.factor_upsample[1] - 0.5
            # x_index = tf.linspace(self.factor_upsample[0]/2 + 0.5, len(x_index)*self.factor_upsample[0] - algo,
            # len(x_index))
        return x_index, y_index  # Return the big index

    @staticmethod
    def _get_4_closest_points(x, y, x_list, y_list):
        # This function gets the 4 closests points
        x_distance = x_list - x
        y_distance = y_list - y
        if tf.math.reduce_any(x_distance >= 0):
            x_min = tf.math.reduce_min(tf.boolean_mask(x_distance, x_distance >= 0))
            x2 = tf.where(x_min == x_distance)[0][0]
        else:
            x2 = -1
        if tf.math.reduce_any(x_distance <= 0):
            x_min = tf.math.reduce_min(tf.math.abs(tf.boolean_mask(x_distance, x_distance <= 0)))
            x1 = tf.where(-x_min == x_distance)[0][0]
        else:
            x1 = -1
        if tf.math.reduce_any(y_distance >= 0):
            y_min = tf.math.reduce_min(tf.boolean_mask(y_distance, y_distance >= 0))
            y2 = tf.where(y_min == y_distance)[0][0]
        else:
            y2 = -1
        if tf.math.reduce_any(y_distance <= 0):
            y_min = tf.math.reduce_min(tf.math.abs(tf.boolean_mask(y_distance, y_distance <= 0)))
            y1 = tf.where(-y_min == y_distance)[0][0]
        else:
            y1 = -1
        if x2 == -1:
            x2 = x1
        if x1 == -1:
            x1 = x2
        if y2 == -1:
            y2 = y1
        if y1 == -1:
            y1 = y2
        return x_list[x2], x_list[x1], y_list[y2], y_list[y1]

    def get_real_equivalent(self):
        return ComplexUpSampling2D(size=self.factor_upsample, data_format=self.data_format,
                                   interpolation=self.interpolation, dtype=self.my_dtype.real_dtype)


