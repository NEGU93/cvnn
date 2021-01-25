Glorot Uniform
--------------

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

.. [GLOROT-2010] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010.