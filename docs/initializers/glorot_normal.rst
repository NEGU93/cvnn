Glorot Normal
-------------

.. py:class:: GlorotNormal(RandomInitializer)

    The Glorot normal initializer, also called Xavier normal initializer.
    
    Reference: [GLOROT-2010]_

    .. note:: The reference actually refers to the uniform case but it's analysis was adapted for a normal distribution
    
    Draws samples from a truncated normal distribution centered on 0 with
    
    * Real case: :code:`stddev = sqrt(2 / (fan_in + fan_out))`
    * Complex case: real part stddev = complex part stddev = :code:`1 / sqrt(fan_in + fan_out)`
    
    where :code:`fan_in` is the number of input units in the weight tensor and :code:`fan_out` is the number of output units.

    Standalone usage::

        import cvnn
        initializer = cvnn.initializers.GlorotNormal()
        values = initializer(shape=(2, 2))                  # Returns a complex Glorot Normal tensor of shape (2, 2)

    Usage in a cvnn layer::

        import cvnn
        initializer = cvnn.initializers.GlorotNormal()
        layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)

.. py:method:: __call__(self, shape, dtype=tf.dtypes.complex64)

    Returns a tensor object initialized as specified by the initializer.

    :param shape: Shape of the tensor.
    :param dtype: Optinal dtype of the tensor. Either floating or complex. ex: :code:`tf.complex64` or :code:`tf.float32`
