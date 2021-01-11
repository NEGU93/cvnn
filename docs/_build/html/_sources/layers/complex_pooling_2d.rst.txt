Complex Pooling 2D
------------------

.. py:class:: ComplexPooling2D

    Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
    Abstract class. This class only exists for code reuse. It will never be an exposed API. 

.. py:method:: __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs)

    :param pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window.
        Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    :param data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        - `channels_last` corresponds to inputs with shape
        - `(batch, height, width, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, height, width)`.
    :param name: A string, the name of the layer.


Complex Max Pooling 2D
^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: ComplexMaxPooling2D

    Max pooling operation for 2D spatial data.
    Works for complex dtype using the absolute value to get the max.

.. warning:: 
    ATTENTION: This layer seams to be doing a strange casting to real when implemented in a Sequential model. It is therefore not correctly working for the moment.

**Complex dtype example**

First, let's create a complex image

.. code-block:: python

    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 4, 5],
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
    print(img[...,0])

This outputs

.. code-block:: python

    [
        [0.+0.j 1.+4.j 2.+5.j]
        [0.+3.j 2.+7.j 2.+9.j]
        [0.+4.j 5.+5.j 7.+3.j]
    ],[
        [0.+0.j 1.+4.j 2.+5.j]
        [0.+3.j 2.+7.j 2.+9.j]
        [0.+4.j 5.+5.j 7.+3.j]
    ]

Now let's run the :code:`ComplexMaxPooling2D` layer

.. code-block:: python

    max_pool = ComplexMaxPooling2D(strides=1)
    res = max_pool(img.astype(np.complex64))
    print(res[...,0])

The results is then

.. code-block:: python

    <tf.Tensor: shape=(2, 2, 2), dtype=complex64, numpy=
    array([[
            [2.+7.j, 2.+9.j],
            [2.+7.j, 2.+9.j]
        ],[
            [7.+2.j, 9.+2.j],
            [5.+8.j, 3.+9.j]
        ]], dtype=complex64)>

**Real dtype example**

This layer also works for real-valued input, for example:

.. code-block:: python

    x = tf.constant([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]])
    x = tf.reshape(x, [1, 3, 3, 1])
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
    complex_max_pool_2d = ComplexMaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
    assert np.all(max_pool_2d(x) == complex_max_pool_2d(x))


Complex Average Pooling 2D
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: ComplexAvgPooling2D

    Average pooling operation for spatial data.
    Works for complex and real dtype.

**Complex dtype example**

First, let's create a complex image

.. code-block:: python

    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 4, 5],
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
    print(img[...,0])

This outputs

.. code-block:: python

    [
        [0.+0.j 1.+4.j 2.+5.j]
        [0.+3.j 2.+7.j 2.+9.j]
        [0.+4.j 5.+5.j 7.+3.j]
    ],[
        [0.+0.j 1.+4.j 2.+5.j]
        [0.+3.j 2.+7.j 2.+9.j]
        [0.+4.j 5.+5.j 7.+3.j]
    ]

Now let's run the :code:`ComplexAvgPooling2D` layer

.. code-block:: python

    avg_pool = ComplexAvgPooling2D(strides=1)
    res = avg_pool(img.astype(np.complex64))
    print(res[...,0])

The results is then

.. code-block:: python

    tf.Tensor([[
        [0.75+3.5j  1.75+6.25j]
        [1.75+4.75j 4.  +6.j  ]
    ],[
        [3.5 +2.25j 6.25+3.25j]
        [4.75+4.25j 6.  +5.25j]
    ]], shape=(2, 2, 2), dtype=complex64)
