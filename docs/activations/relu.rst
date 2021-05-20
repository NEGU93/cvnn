ReLU-based
----------

.. py:method:: modrelu(z: Tensor, b: float, c: float = 1e-3)
    
    mod ReLU presented in [CIT2016-ARJOVSKY]_.
    
    A variation of the ReLU named modReLU. It is a pointwise nonlinearity,
    :math:`modReLU(z) : C \longrightarrow C`, which affects only the absolute
    value of a complex number, defined
    
    .. math::
    
        modReLU(z) = ReLU(|z|+b)*z/|z|

.. py:method:: crelu(z: Tensor)

    Mirror of :code:`cvnn.activations.cart_relu`.
    Applies `Rectified Linear Unit <https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu>`_ to both the real and imag part of z.

    The relu function, with default values, it returns element-wise max(x, 0).

    Otherwise, it follows.

    .. math::

        f(x) = \textrm{max_value}, \quad \textrm{for} \quad x >= \textrm{max_value} \\
        f(x) = x, \quad \textrm{for} \quad \textrm{threshold} <= x < \textrm{max_value} \\
        f(x) = \alpha * (x - \textrm{threshold}), \quad \textrm{otherwise} \\

    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: zrelu(z: Tensor)

    zReLU presented in [CIT2016-GUBERMAN]_.
    This methods let's the output as the input if both real and imaginary parts are positive.
    
    .. math::
    
        f(z) = z \quad \textrm{for} \quad 0 \leq \phi_z \leq \pi / 2 \\
        f(z) =    0 \quad \textrm{elsewhere}  \\

.. py:method:: complex_cardioid(z: Tensor)

    Complex cardioid presented in [CIT2017-PATRICK]_

    This function maintains the phase information while attenuating the magnitude based on the phase itself. 
    For real-valued inputs, it reduces to the ReLU.

    .. math::

        f(z) = \frac{(1 + cos \phi_z) * z}{2}


             
             
.. [CIT2016-ARJOVSKY] M. Arjovsky et al. "Unitary Evolution Recurrent Neural Networks" 2016
.. [CIT2016-GUBERMAN] N. Guberman "On Complex Valued Convolutional Neural Networks" 2016
.. [CIT2017-PATRICK] V. Patrick et al. "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting" 2017
