Complex input, real output
--------------------------

.. py:method:: softmax_real(z, axis=-1)

    Applies the `softmax <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax>`_ function to the modulus of z.
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    The softmax of x is calculated by:

    .. math::
	
		\frac{e^x}{\textrm{tf.reduce_sum}(e^x)}
    

    :param z: Input tensor.
    :return: Real-valued tensor of the applied activation function


.. py:method:: convert_to_real_with_abs(z)

    Applies the absolute value and returns a real-valued output.

    :param z: Input tensor.
    :return: Real-valued tensor of the applied activation function