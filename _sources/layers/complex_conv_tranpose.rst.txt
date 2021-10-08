Complex Convolution 2D Transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: ComplexConv2DTranspose

    Complex Transposed convolution layer. Sometimes (wrongly) called Deconvolution.

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

.. py:method:: __init__(self, filters, kernel_size, strides, padding, dtype, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)

    :param filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any :code:`dilation_rate` value != 1.
    :param padding: one of :code:`"valid"` or :code:`"same"` (case-insensitive).
      :code:`"valid"` means no padding. :code:`"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
    :param output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width of the output tensor.
      Can be a single integer to specify the same value for all spatial dimensions.
      The amount of output padding along a given dimension must be lower than the stride along that same dimension.
      If set to :code:`None` (default), the output shape is inferred.
    :param data_format: A string,
      one of `:code:channels_last` (default) or :code:`channels_first`.
      The ordering of the dimensions in the inputs.
      :code:`channels_last` corresponds to inputs with shape :code:`(batch_size, height, width, channels)` while :code:`channels_first`
      corresponds to inputs with shape :code:`(batch_size, channels, height, width)`.
      It defaults to the :code:`image_data_format` value found in your Keras config file at :code:`~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    :param dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is incompatible with specifying any stride value != 1.
    :param activation: Activation function to use.
      If you don't specify anything, no activation is applied (see `keras.activations`).
    :param use_bias: Boolean, whether the layer uses a bias vector.
    :param kernel_initializer: Initializer for the `kernel` weights matrix (see `keras.initializers`).
    :param bias_initializer: Initializer for the bias vector (see `keras.initializers`).
    :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix (see `keras.regularizers`).
    :param bias_regularizer: Regularizer function applied to the bias vector (see `keras.regularizers`).
    :param activity_regularizer: Regularizer function applied to the output of the layer (its "activation") (see `keras.regularizers`).
    :param kernel_constraint: Constraint function applied to the kernel matrix (see `keras.constraints`).
    :param bias_constraint: Constraint function applied to the bias vector (see `keras.constraints`).