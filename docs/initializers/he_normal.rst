He Normal
---------

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


.. [HE-2015] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.