Un-pooling 2D
^^^^^^^^^^^^^

.. warning::

    Still has a bug, if argmax has coincident indexes. Don't know if this is a desired behaivour or not.

.. py:class:: ComplexUnPooling2D

    This class was inspired to recreate the CV-FCN model of [CIT2019-CAO]_

    Performs UnPooling as explained `here <https://www.oreilly.com/library/view/hands-on-convolutional-neural/9781789130331/6476c4d5-19f2-455f-8590-c6f99504b7a5.xhtml>`_.

    There are 2 main ways to use this function, either give the expected output shape or give an upsampling_factor. The second options is the only way to deal with partially known output, for example (None, None, 3) to deal with variable size iamges.

.. figure:: ../_static/max_unpool_explain.png

**Usage example with desired_output_shape**

.. code-block:: python

    from cvnn.layers import ComplexUnPooling2D, complex_input, ComplexMaxPooling2DWithArgmax
    import tensorflow as tf
    import numpy as np

    def get_img():
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
        return img

    x = get_img()       # Gets an image just for the example
    inputs = complex_input(shape=x.shape[1:])
    # Apply max-pooling and gets also argmax
    max_pool_o, max_arg = ComplexMaxPooling2DWithArgmax(strides=1, data_format="channels_last", name="argmax")(inputs)
    # Applies the Unpooling
    outputs = ComplexUnPooling2D(x.shape[1:])([max_pool_o, max_arg])
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pooling_model")
    model.summary()
    model(x)

**Usage example with upsampling_factor**

    inputs = complex_input(shape=(None, None, 3))  # Input is an unknown size RGB image
    max_pool_o, max_arg = ComplexMaxPooling2DWithArgmax(strides=1, data_format="channels_last", name="argmax")(inputs)
    unpool = ComplexUnPooling2D(upsampling_factor=2)([input_to_block, pool_argmax])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pooling_model")
    model.summary()
    model(x)


.. py:method:: __init__(self, desired_output_shape, name=None, dtype=DEFAULT_COMPLEX_TYPE, dynamic=False, **kwargs)

    :param desired_output_shape: tf.TensorShape (or equivalent like tuple or list). The expected output shape without the batch size. Meaning that for a 2D image to be enlarged, this is size 3 of the form HxWxC or CxHxW
    :param upsampling_factor: Integer. The factor to which enlarge the image. For example, if upsampling_factor=2, an input image of size 32x32 will be 64x64. This parameter is ignored if desired_output_shape is used or if the output shape is given to the call funcion.

.. py:method:: call(self, inputs, **kwargs)

    :param inputs: A tuple of Tensor objects :code:`(input, argmax)`.

        - :code:`input` A Tensor.
        - :code:`argmax` A Tensor. The indices in argmax are flattened (Complains directly to TensorFlow)



.. [CIT2019-CAO] Cao, Wu, Zhang, Liang and Li. "Pixel-Wise PolSAR Image Classification via a Novel Complex-Valued Deep Fully Convolutional Network" DPI: 10.3390/rs11222653, 2019.
