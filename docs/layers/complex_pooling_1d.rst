Complex Pooling 1D
^^^^^^^^^^^^^^^^^^

.. py:class:: ComplexPooling2D

    Pooling layer for arbitrary pooling functions, for 1D inputs.
    Abstract class. This class only exists for code reuse. It will never be an exposed API. 

.. py:method:: __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, name=None, **kwargs)

    :param pool_size: An integer: Specifying the size of the pooling window.
    :param strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
    :param padding: One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
    :param data_format: A string, one of `channels_last` (default) or `channels_first`. The ordering of the dimensions in the inputs. :code:`channels_last` corresponds to inputs with shape :code:`(batch, steps, features)` while :code:`channels_first` corresponds to inputs with shape :code:`(batch, features, steps)`.
    :param name: A string, the name of the layer.

.. _complex-max-pooling-label:

Complex Average Pooling 1D
""""""""""""""""""""""""""

.. py:class:: ComplexAvgPooling1D

    Downsamples the input representation by taking the average value over the window defined by :code:`pool_size`. 
    The window is shifted by strides. The resulting output when using "valid" padding option has a shape of: :code:`output_shape = (input_shape - pool_size + 1) / strides)`
    The resulting output shape when using the "same" padding option is: :code:`output_shape = input_shape / strides`

**Complex dtype example**

First, let's create a complex image

.. code-block:: python

    img_r = np.array([[
        [0, 1, 2, 0, 2, 2, 0, 5, 7]
    ], [
        [0, 4, 5, 3, 7, 9, 4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5, 3, 7, 9, 4, 5, 3]
    ], [
        [0, 4, 5, 3, 2, 2, 4, 8, 9]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i
    img = np.reshape(img, (2, 9, 1))
    print(img[...,0])

This outputs

.. code-block:: python

    [[0.+0.j 1.+4.j 2.+5.j 0.+3.j 2.+7.j 2.+9.j 0.+4.j 5.+5.j 7.+3.j]
     [0.+0.j 4.+4.j 5.+5.j 3.+3.j 7.+2.j 9.+2.j 4.+4.j 5.+8.j 3.+9.j]]

Now let's run the :code:`ComplexAvgPooling1D` layer

.. code-block:: python

    avg_pool = ComplexAvgPooling1D(strides=1)
    res = avg_pool(img.astype(np.complex64))
    print(res[...,0])

The results is then

.. code-block:: python

    tf.Tensor(
                [[0.5+2.j  1. +4.j  2. +8.j  2.5+4.5j]
                 [2. +2.j  4. +4.j  8. +2.j  4.5+6.j ]], 
              shape=(2, 4), dtype=complex64)