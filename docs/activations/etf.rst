Elementary Transcentental Functions
-----------------------------------

These types of activation functions where highly explored by Taehwan Kim and Tulay Adali, mainly in [CIT2001-KIM]_ and [CIT2003-KIM]_. 
Please refer to them for further information. These functions are divided into 4 groups.

- Circular
- Inverse Circular
- Hyperbolic
- Inverse Hyperbolic

Circular
^^^^^^^^

.. py:method:: etf_circular_tan(z):

    Computes tan of z element-wise.

    .. math::

        tan(z) = \frac{e^{iz} - e^{-iz}}{i(e^{iz} + e^{-iz})}

    .. raw:: html

        <object data="../_static/etf/tan.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

.. py:method:: etf_circular_sin(z):

    Computes sine of z element-wise.

    .. math::

        sin(z) = \frac{e^{iz} - e^{-iz}}{2i}

    .. raw:: html

        <object data="../_static/etf/sin.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

Inverse Circular
^^^^^^^^^^^^^^^^

.. py:method:: etf_inv_circular_atan(z):

    Computes the trignometric inverse tangent of z element-wise.

    .. math::

        atan(z) = \int_{0}^{z} \frac{dt}{1+t^2}

    .. raw:: html

        <object data="../_static/etf/atan.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

.. py:method:: etf_inv_circular_asin(z):

    Computes the trignometric inverse sine of z element-wise.

    .. math::

        asin(z) = \int_{0}^{z} \frac{dt}{(1-t)^1/2}

    .. raw:: html

        <object data="../_static/etf/asin.png" type="image/png"></object>
    

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

.. py:method:: etf_inv_circular_acos(z):

    Computes acos of z element-wise.

    .. math::

        acos(z) = \int_{z}^{1} \frac{dt}{(1-t^2)^1/2}

    .. raw:: html

        <object data="../_static/etf/acos.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

Hyperbolic
^^^^^^^^^^

.. py:method:: etf_circular_tanh(z):

    Computes hyperbolic tangent of z element-wise.

    .. math::

        tanh(z) = \frac{sinh(z)}{cosh(z)} = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}

    .. raw:: html

        <object data="../_static/etf/tanh.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

.. py:method:: etf_circular_sinh(z):

    Computes hyperbolic sine of z element-wise.

    .. math::

        sinh(z) = \frac{e^{z} - e^{-z}}{2}

    .. raw:: html

        <object data="../_static/etf/sinh.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

Inverse Hyperbolic
^^^^^^^^^^^^^^^^^^

.. py:method:: etf_inv_circular_atanh(z):

    Computes inverse hyperbolic tangent of z element-wise.

    .. math::

        atanh(z) = \int_{0}^{z} \frac{dt}{1-t^2}

    .. raw:: html

        <object data="../_static/etf/atanh.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.

.. py:method:: etf_inv_circular_asinh(z):

    Computes inverse hyperbolic sine of z element-wise.

    .. math::

        asinh(z) = \int_{0}^{z} \frac{dt}{(1+t^2)^1/2}

    .. raw:: html

        <object data="../_static/etf/asinh.png" type="image/png"></object>

    :param z:	A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int8, int16, int32, int64, complex64, complex128.



.. [CIT2001-KIM] T. Kim and T Adali "Complex Backpropagation neural network using elementary transdencental activation functions" 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No.01CH37221)

.. [CIT2003-KIM] T. Kim and T Adali "Approximation by Fully Complex MLP Using Elementary Transcendental Activation Functions" 2001 Neural Computation
