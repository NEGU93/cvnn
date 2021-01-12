Activation Functions
====================

.. _activations:

It is recommended to only use the activation functions using::

	apply_activation(act_fun, out)

.. note:: You can define your own activation function, just create it inside the :code:`activation_function.py` and add it to :code:`act_dispatcher`

.. note:: The following :code:`cart` or :code:`pol` means either type A (cartesian) or type B (polar) according to [CIT2003-KUROE]_ notation.

.. py:method:: apply_activation(act_fun, out)

	Applies activation function :code:`act` to variable :code:`out`

    :param out: Tensor to whom the activation function will be applied
    :param act_fun: function to be applied to out. 
    :return: Tensor with the applied activation function

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
   
.. py:method:: linear(z)

	Does not apply any activation function. It just outputs the input.
	
    :param z: Input tensor variable
    :return: z

TYPE A: Cartesian form
----------------------

.. py:method:: cart_sigmoid(z)

	Called with :code:`'cart_sigmoid'` string. 

	Applies the `sigmoid <https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid>`_ function to both the real and imag part of z.

	.. math::

		\frac{1}{1 + e^{-x}} + j  \frac{1}{1 + e^{-y}}

	where 

	.. math::

		z = x + j y

    :param z: Tensor to be used as input of the activation function
    :return: Tensor result of the applied activation function

.. py:method:: cart_elu(x, alpha=0.1)

	Applies the `Exponential linear unit <https://www.tensorflow.org/api_docs/python/tf/keras/activations/elu>`_. To both the real and imaginary part of z.
    
	.. math::
	
		x if x > 0 and alpha * (exp(x)-1) if x < 0

    :param z: Input tensor.
    :param alpha: A scalar, slope of negative section.
    :return: Tensor result of the applied activation function

.. py:method:: cart_exponential(z)

	Exponential activation function. Applies to both the real and imag part of z the `exponential activation <https://www.tensorflow.org/api_docs/python/tf/keras/activations/exponential>`_:
	 
	
	.. math::
		e^x
    

    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: cart_hard_sigmoid(z)

	Applies the `hard Sigmoid <https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid>`_ function to both the real and imag part of z.
    The hard sigmoid function is faster to compute than sigmoid activation.
    Hard sigmoid activation:    

	.. math::

			0               ,\quad x < -2.5 \\
            1               ,\quad x > 2.5 \\
            0.2 * x + 0.5   ,\quad -2.5 <= x <= 2.5
    

    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: cart_relu(z, , alpha=0.0, max_value=None, threshold=0)

	Applies `Rectified Linear Unit <https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu>`_ to both the real and imag part of z.

    The relu function, with default values, it returns element-wise max(x, 0).

    Otherwise, it follows:  
	
	.. math::

			f(x) = \textrm{max_value}, \quad \textrm{for} \quad x >= \textrm{max_value} \\
            f(x) = x, \quad \textrm{for} \quad \textrm{threshold} <= x < \textrm{max_value} \\
            f(x) = \alpha * (x - \textrm{threshold}), \quad \textrm{otherwise} \\
    
    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: cart_leaky_relu(z, alpha=0.2, name=None)

	Applies `Leaky Rectified Linear Unit <https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu>`_ [CIT2013-MAAS]_ (`source <http://robotics.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_) to both the real and imag part of z.

    :param z: Input tensor.
    :param alpha: Slope of the activation function at x < 0. Default: 0.2
    :param name: A name for the operation (optional).
    :return: Tensor result of the applied activation function

.. py:method:: cart_selu(z)

    Applies `Scaled Exponential Linear Unit (SELU) <https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu>`_ [CIT2017-KLAMBAUER]_ (`source <https://arxiv.org/abs/1706.02515>`_) to both the real and imag part of z.
    
    

    The scaled exponential unit activation:

    .. math::
        \textrm{scale} * \textrm{elu}(x, \alpha)
    

    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: cart_softplus(z):

    Applies `Softplus <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus>`_ activation function to both the real and imag part of z.
    The Softplus function: 
    
    .. math::
        log(e^x + 1)
    
    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: cart_softsign(z):
    
    Applies `Softsign <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softsign>`_ activation function to both the real and imag part of z.
    The softsign activation: 
    
    .. math::

        \frac{x}{\lvert x \rvert + 1}    

    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. py:method:: cart_tanh(z)

	Applies `Hyperbolic Tangent <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>`_ (tanh) activation function to both the real and imag part of z.

    The tanh activation: 
	
	.. math::

		tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}.

    The derivative if tanh is computed as  :math:`1 - tanh^2` so it should be fast to compute for backprop.
    
    :param z: Input tensor.
    :return: Tensor result of the applied activation function

Classification
^^^^^^^^^^^^^^

.. py:method:: cart_softmax(z, axis=-1)

	Applies the `softmax function <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax>`_ to both the real and imag part of z.
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    The softmax of x is calculated by:
	
	.. math::
	
		\frac{e^x}{\textrm{tf.reduce_sum}(e^x)}
    
    :param z: Input tensor.
    :return: Tensor result of the applied activation function


TYPE B: Polar form
------------------

.. py:method:: pol_selu(z)

    Applies `Scaled Exponential Linear Unit (SELU) <https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu>`_ [CIT2017-KLAMBAUER]_ (`source <https://arxiv.org/abs/1706.02515>`_) to the absolute value of z, keeping the phase unchanged.

    The scaled exponential unit activation:

    .. math::

        \textrm{scale} * \textrm{elu}(x, \alpha)
    
    :param z: Input tensor.
    :return: Tensor result of the applied activation function

.. [CIT2003-KUROE] Kuroe, Yasuaki, Mitsuo Yoshid, and Takehiro Mori. "On activation functions for complex-valued neural networks—existence of energy functions—." Artificial Neural Networks and Neural Information Processing—ICANN/ICONIP 2003. Springer, Berlin, Heidelberg, 2003. 985-992.

.. [CIT2013-MAAS] A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectifier Nonlinearities Improve Neural Network Acoustic Models,” 2013.

.. [CIT2017-KLAMBAUER] G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter, “Self-Normalizing Neural Networks,” ArXiv170602515 Cs Stat, Sep. 2017. Available: http://arxiv.org/abs/1706.02515.

