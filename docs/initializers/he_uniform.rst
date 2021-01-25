He Uniform
----------

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