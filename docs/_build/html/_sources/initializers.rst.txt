Initializers
============

.. py:class:: RandomInitializer

    Random initializer helps generate a random tensor of

    * Either complex or real (floating) data type
    * Either Uniform or Normal distribution
    * Zero mean

    How to use it::

        # creates a complex tensor of shape (3, 3) distribution with Re{random_tensor} ~ U[-2, 2] and Im{random_tensor} ~ U[-3, 3]
        random_tensor = RandomInitializer(distribution="uniform")(shape=(3, 3), c_limit=[2, 3], dtype=tf.complex)

.. py:method:: __init__(self, distribution="uniform", seed=None)

    :param distribution: It can be either a uniform or a normal distribution.
    :param seed: Integer. Used to create a random seed for the distribution. See :code:`tf.random.set_seed`.

.. py:method:: get_random_tensor(self, shape, c_arg=None, r_arg=None, dtype=tf.dtypes.complex64)

    Outputs random values either uniform or normal according to initialization
    
    :param shape: The shape of the output tensor.
    :param r_arg: Argument.
        If uniform, the output will be a distribution between [-arg, arg].
        If Normal, the output will be a zero-mean gaussian distribution with arg stddev
    :param c_arg: Tuple of the argument for the real and imaginary part respectively.
    :param dtype: The type of the output. Default :code:`tf.complex`.

    .. note:: Either :code:`c_arg` or :code:`r_arg` will be used according to :code:`dtype` parameter, the one not being used will be ignored and can be set to :code:`None` (default).


.. py:class:: GlorotUniform(RandomInitializer)

    The Glorot uniform initializer, also called Xavier uniform initializer.
    
    Reference: [GLOROT-2010]_

    Draws samples from a uniform distribution:
    
    * Real case: :code:`x ~ U[-limit, limit]` where :code:`limit = sqrt(6 / (fan_in + fan_out))`
    * Complex case: :code:`z / Re{z} = Im{z} ~ U[-limit, limit]` where :code:`limit = sqrt(3 / (fan_in + fan_out))`
    where :code:`fan_in` is the number of input units in the weight tensor and :code:`fan_out` is the number of output units.

    Standalone usage::

        import cvnn
        initializer = cvnn.initializers.GlorotUniform()
        values = initializer(shape=(2, 2))                  # Returns a complex Glorot Uniform tensor of shape (2, 2)

    Usage in a cvnn layer::

        import cvnn
        initializer = cvnn.initializers.GlorotUniform()
        layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)

.. py:method:: __init__(self, seed=None, scale=1.)
    
    :param seed: Integer. An initializer created with a given seed will always produce the same random tensor for a given shape and dtype.
    :param scale: Default 1. Scales the limit as :code:`limit = scale * limit`

.. py:method:: __call__(self, shape, dtype=tf.dtypes.complex64)
        
    Returns a tensor object initialized as specified by the initializer.

    :param shape: Shape of the tensor.
    :param dtype: Optional dtype of the tensor. Either floating or complex. ex: :code:`tf.complex64` or :code:`tf.float32`

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

.. py:class:: HeNormal(RandomInitializer)

    He normal initializer.

    Reference: [HE-2015]_
    
    It draws samples from a truncated normal distribution centered on 0 with
    
    * Real case: :code:`stddev = sqrt(2 / fan_in)`
    * Complex case: real part stddev = complex part stddev = :code:`1 / sqrt(fan_in)`
    
    where :code:`fan_in` is the number of input units in the weight tensor.

    Standalone usage::

        import cvnn
        initializer = cvnn.initializers.HeNormal()
        values = initializer(shape=(2, 2))                  # Returns a complex He Normal tensor of shape (2, 2)
    
    
    Usage in a cvnn layer::

        import cvnn
        initializer = cvnn.initializers.HeNormal()
        layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)

.. py:method:: __call__(self, shape, dtype=tf.dtypes.complex64)

    Returns a tensor object initialized as specified by the initializer.

    :param shape: Shape of the tensor.
    :param dtype: Optinal dtype of the tensor. Either floating or complex. ex: :code:`tf.complex64` or :code:`tf.float32`


.. py:class:: HeUniform(RandomInitializer)

    The He Uniform initializer.

    Reference: [HE-2015]_

    Draws samples from a uniform distribution

    * Real case: :code:`x ~ U[-limit, limit]` where :code:`limit = sqrt(6 / fan_in)`
    * Complex case: :code:`z / Re{z} = Im{z} ~ U[-limit, limit]` where :code:`limit = sqrt(3 / fan_in)`
    where :code:`fan_in` is the number of input units in the weight tensor.

    
    Standalone usage::

        import cvnn
        initializer = cvnn.initializers.HeUniform()
        values = initializer(shape=(2, 2))                  # Returns a complex He Uniform tensor of shape (2, 2)
    
    
    Usage in a cvnn layer::

        import cvnn
        initializer = cvnn.initializers.HeUniform()
        layer = cvnn.layers.Dense(input_size=23, output_size=45, weight_initializer=initializer)


.. py:method:: __call__(self, shape, dtype=tf.dtypes.complex64)

    Returns a tensor object initialized as specified by the initializer.

    :param shape: Shape of the tensor.
    :param dtype: Optinal dtype of the tensor. Either floating or complex. ex: :code:`tf.complex64` or :code:`tf.float32`

.. [GLOROT-2010] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010.

.. [HE-2015] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.
