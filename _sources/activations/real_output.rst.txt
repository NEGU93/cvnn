Complex input, real output
--------------------------

.. py:method:: convert_to_real_with_abs(z)

Applies the absolute value and returns a real-valued output.

:param z: Input tensor.
:return: Real-valued tensor of the applied activation function

Softmax Based
^^^^^^^^^^^^^

The following function will always output a real-valued output even if the input is complex.

All the functions use the `softmax <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax>`_ function as a base.
If the input is real-valued they all apply the conventional softmax function to the data.
The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
It is often used as the activation for the last layer of a classification network because the result could be
interpreted as a probability distribution.
The softmax of x is calculated by:

.. math::

  \sigma = \frac{e^x}{\textrm{tf.reduce_sum}(e^x)}


.. py:method:: softmax_real_with_abs(z, axis=-1)

    Applies the  function to the modulus of z (only if z is complex).

    .. math::

      out = \sigma(|z|)
    
    :param z: Input tensor.
    :param axis:	(Optional) Integer, axis along which the softmax normalization is applied.
    :return: Real-valued tensor of the applied activation function

.. py:method:: softmax_real_with_avg(z, axis=-1)

    Applies the function to the real and imaginary part of z separately and then averages it.

    .. math::

      out = \frac{\sigma(x) + \sigma(y)}{2}
        
    :param z: Input tensor.
    :param axis:	(Optional) Integer, axis along which the softmax normalization is applied.
    :return: Real-valued tensor of the applied activation function

.. py:method:: softmax_real_with_mult(z, axis=-1)

    Applies the function to the real and imaginary part of z separately and then multiplies them.

    .. math::

      out = \sigma(x) * \sigma(y)
            
    :param z: Input tensor.
    :param axis:	(Optional) Integer, axis along which the softmax normalization is applied.
    :return: Real-valued tensor of the applied activation function

.. py:method:: softmax_of_softmax_real_with_mult(z, axis=-1)

    Applies the function to the real and imaginary part of z separately and then applies the function again on the product of them.

    .. math::

      out = \sigma(\sigma(x) * \sigma(y))
                
    :param z: Input tensor.
    :param axis:	(Optional) Integer, axis along which the softmax normalization is applied.
    :return: Real-valued tensor of the applied activation function

.. py:method:: softmax_of_softmax_real_with_avg(z, axis=-1)

    Applies the function to the real and imaginary part of z separately and then applies the function again on the sum of them.

    .. math::

      out = \sigma(\sigma(x) + \sigma(y))
                
    :param z: Input tensor.
    :param axis:	(Optional) Integer, axis along which the softmax normalization is applied.
    :return: Real-valued tensor of the applied activation function


.. py:method:: softmax_real_with_polar(z, axis=-1)

    Applies the function to the amplitude and phase of z separately and then averages them.

    .. math::

      out = \frac{\sigma(|z|) + \sigma(\phi_z))}{2}
                
    :param z: Input tensor.
    :param axis:	(Optional) Integer, axis along which the softmax normalization is applied.
    :return: Real-valued tensor of the applied activation function
