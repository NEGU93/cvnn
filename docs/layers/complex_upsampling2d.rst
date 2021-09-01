Complex Upsampling 2D
^^^^^^^^^^^^^^^^^^^^^



.. py:class:: ComplexUpSampling2D

    Upsampling layer for 2D inputs.

    The algorithms available are nearest neighbor or bilinear.

    **Usage example**

.. code-block:: python

    import tensorflow as tf
    from cvnn.layers import ComplexUnPooling2D
    x = tf.convert_to_tensor([[[[1., 2.], [3., 4.]]]])
    z = tf.complex(real=x, imag=x)
    y_tf = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')(x)
    y_cvnn = ComplexUpSampling2D(size=2, interpolation='bilinear', data_format='channels_first')(z)
    assert np.all(y_tf == tf.math.real(y_cvnn).numpy())

.. py:method:: __init__(self, size=(2, 2), data_format: Optional[str] = None, interpolation: str = 'nearest', dtype=DEFAULT_COMPLEX_TYPE, **kwargs)

    :param size: Int, or tuple of 2 integers. The upsampling factors for rows and columns.
    :param data_format: string, one of :code:`channels_last` (default) or :code:`channels_first`. The ordering of the dimensions in the inputs. :code:`channels_last` corresponds to inputs with shape :code:`(batch_size, height, width, channels)` while :code:`channels_first` corresponds to inputs with shape :code:`(batch_size, channels, height, width)`.
    :param interpolation: A string, one of :code:`nearest` or :code:`bilinear`.
..    :param align_corners:  if :code:`True`, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. An example is shown in the following picture.

..    .. figure:: ../_static/align_corners.png