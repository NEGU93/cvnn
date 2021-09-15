import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import UpSampling2D
from typing import Optional, Union, Tuple
from cvnn.layers.core import ComplexLayer
from cvnn.layers.core import DEFAULT_COMPLEX_TYPE


class ComplexUpSampling2D(UpSampling2D, ComplexLayer):

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
            Example of align corners: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        """
        self.factor_upsample = size
        self.my_dtype = tf.dtypes.as_dtype(dtype)
        super(ComplexUpSampling2D, self).__init__(size=size, data_format=data_format, interpolation=interpolation,
                                                  dtype=self.my_dtype.real_dtype, **kwargs)

    def call(self, inputs):
        result = tf.complex(
            backend.resize_images(tf.math.real(inputs), self.size[0], self.size[1], self.data_format,
                                  interpolation=self.interpolation),
            backend.resize_images(tf.math.imag(inputs), self.size[0], self.size[1], self.data_format,
                                  interpolation=self.interpolation),
        )
        casted_value = inputs.dtype if not inputs.dtype.is_integer else tf.float32
        return tf.cast(result, dtype=casted_value)

    def get_real_equivalent(self):
        return ComplexUpSampling2D(size=self.factor_upsample, data_format=self.data_format,
                                   interpolation=self.interpolation, dtype=self.my_dtype.real_dtype)

    def get_config(self):
        config = super(ComplexUpSampling2D, self).get_config()
        config.update({
            'dtype': self.my_dtype,
            'factor_upsample': self.factor_upsample

        })
        return config


if __name__ == "__main__":
    image = tf.constant([
        [1., 0., 0, 0, 0],
        [0, 1., 0, 0, 0],
        [0, 0, 1., 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    image = tf.complex(image, image)
    image = image[tf.newaxis, ..., tf.newaxis]
    result = ComplexUpSampling2D([3, 5])(image)
    import pdb; pdb.set_trace()

