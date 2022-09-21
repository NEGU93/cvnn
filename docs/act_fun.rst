Activation Functions
====================

.. _activations:

.. toctree::
	:maxdepth: 2

    activations/types
    activations/real_output
    activations/relu
    activations/mvn_activation.ipynb
    activations/etf


There are two ways to use an activation function

Option 1: Using the string as in :code:`act_dispatcher`::

    ComplexDense(units=x, activation='cart_sigmoid')

Option 2: Using the function directly::

    from cvnn.activations import cart_sigmoid

    ComplexDense(units=x, activation=cart_sigmoid)

.. note:: 
    Unless explicitedly said otherwise, these activation functions does not change the input dtype.

List of activation functions::

    act_dispatcher = {
        'linear': linear,
        # Complex input, real output
        'cast_to_real': cast_to_real,
        'convert_to_real_with_abs': convert_to_real_with_abs,
        'sigmoid_real': sigmoid_real,
        'softmax_real_with_abs': softmax_real_with_abs,
        'softmax_real_with_avg': softmax_real_with_avg,
        'softmax_real_with_mult': softmax_real_with_mult,
        'softmax_of_softmax_real_with_mult': softmax_of_softmax_real_with_mult,
        'softmax_of_softmax_real_with_avg': softmax_of_softmax_real_with_avg,
        'softmax_real_with_polar': softmax_real_with_polar,
        # Phasor networks
        'georgiou_cdbp': georgiou_cdbp,
        'mvn_activation': mvn_activation,
        'complex_signum': complex_signum,
        # Type A (cartesian)
        'cart_sigmoid': cart_sigmoid,
        'cart_elu': cart_elu,
        'cart_exponential': cart_exponential,
        'cart_hard_sigmoid': cart_hard_sigmoid,
        'cart_relu': cart_relu,
        'cart_leaky_relu': cart_leaky_relu,
        'cart_selu': cart_selu,
        'cart_softplus': cart_softplus,
        'cart_softsign': cart_softsign,
        'cart_tanh': cart_tanh,
        'cart_softmax': cart_softmax,
        # Type B (polar)
        'pol_tanh': pol_tanh,
        'pol_sigmoid': pol_sigmoid,
        'pol_selu': pol_selu,
        # Elementary Transcendental Functions (ETF)
        'etf_circular_tan': etf_circular_tan,
        'etf_circular_sin': etf_circular_sin,
        'etf_inv_circular_atan': etf_inv_circular_atan,
        'etf_inv_circular_asin': etf_inv_circular_asin,
        'etf_inv_circular_acos': etf_inv_circular_acos,
        'etf_circular_tanh': etf_circular_tanh,
        'etf_circular_sinh': etf_circular_sinh,
        'etf_inv_circular_atanh': etf_inv_circular_atanh,
        'etf_inv_circular_asinh': etf_inv_circular_asinh,
        # ReLU
        'modrelu': modrelu,
        'crelu': crelu,
        'zrelu': zrelu,
        'complex_cardioid': complex_cardioid
    }


.. note:: To define your own activation function, just create it inside the :code:`activations.py` and add it to :code:`act_dispatcher` dictionary at the end


.. py:method:: linear(z)

	Does not apply any activation function. It just outputs the input.
	
    :param z: Input tensor variable
    :return: z

