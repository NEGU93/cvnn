Complex Convolution
-------------------

Complex Conv 2D
^^^^^^^^^^^^^^^

**Small code example**

.. code-block:: python

    input_shape = (4, 28, 28, 3)
    x = tf.cast(tf.random.normal(input_shape), tf.complex64)
    y = ComplexConv2D(2, 3, activation='cart_relu', padding="same", input_shape=input_shape[1:], dtype=x.dtype)(x)
    assert y.shape == (4, 28, 28, 2)
    assert y.dtype == tf.complex64

.. py:class:: ComplexConv2D

    2D convolution layer (e.g. spatial convolution over images).
    Support complex and real input.
    This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. 
    If :code:`use_bias` is :code:`True`, a bias vector is created and added to the outputs. 
    Finally, if :code:`activation` is not :code:`None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model, provide the keyword argument :code:`input_shape` (tuple of integers, does not include the sample axis),
    e.g. :code:`input_shape=(128, 128, 3)` for 128x128 RGB pictures in :code:`data_format="channels_last"`.


.. py:method:: __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, dtype=np.complex64, kernel_initializer=ComplexGlorotUniform(), bias_initializer=Zeros(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)

    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify  the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. 
        Specifying any stride value != 1 is incompatible with specifying any :code:`dilation_rate` value != 1.
    :param padding: one of :code:`"valid"` or `"same"` (case-insensitive). 
        - :code:`"valid"` means no padding. 
        - :code:`"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
    :param dtype: Dtype of the input and therefore layer. Default :code:`complex64`. 
    :param data_format: A string, one of :code:`channels_last` (default) or :code:`channels_first`.
        The ordering of the dimensions in the inputs. :code:`channels_last` corresponds to inputs with shape :code:`(batch_size, height, width, channels)` while :code:`channels_first` corresponds to inputs with shape :code:`(batch_size, channels, height, width)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. 
        If you never set it, then it will be :code:`channels_last`.
    :param dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. 
        Currently, specifying any :code:`dilation_rate` value != 1 is incompatible with specifying any stride value != 1.
    :param groups: A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately
        with :code:`filters / groups` filters. The output is the concatenation of all the :code:`groups` results along the channel axis. Input channels and :code:`filters` must both be divisible by :code:`groups`.
    :param activation: Activation function to use. If you don't specify anything, no activation is applied (see :code:`keras.activations` or :code:`cvnn.activations`).
        For complex :code:`dtype`, this must be a :code:`cvnn.activations` module.
    :param use_bias: Boolean, whether the layer uses a bias vector.
    :param kernel_initializer: Initializer for the :code:`kernel` weights matrix (see :code:`cvnn.initializers`).
    :param bias_initializer: Initializer for the bias vector (see :code:`cvnn.initializers`).
    :param kernel_regularizer: Regularizer function applied to the :code:`kernel` weights matrix (see :code:`keras.regularizers`).
    :param bias_regularizer: Regularizer function applied to the bias vector (see :code:`keras.regularizers`).
    :param activity_regularizer: Regularizer function applied to the output of the layer (its "activation") (see :code:`keras.regularizers`).
    :param kernel_constraint: Constraint function applied to the kernel matrix (see :code:`keras.constraints`).
    :param bias_constraint: Constraint function applied to the bias vector (see :code:`keras.constraints`).

.. warning:: 
    ATTENTION: :code:`regularizers` not yet working, that parameter will be ignored.

    
.. py:method:: call(self, inputs)

    Calls convolution, this function is divided in 4:
            1. Input parser/verification
            2. Convolution
            3. Bias
            4. Activation Function
    :param inputs: Input tensor, or list/tuple of input tensors.
    :returns: A tensor of rank 4+ representing :code:`activation(conv2d(inputs, kernel) + bias)`.

Complex Conv 1D and 3D
^^^^^^^^^^^^^^^^^^^^^^

This library also has 1D (:code:`ComplexConv1D`) and 3D (:code:`ComplexConv3D`) convolution layers.
Usage is analogous to :code:`ComplexConv2D`.
