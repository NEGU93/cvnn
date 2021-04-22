Activation Functions
====================

.. _activations:


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
        'linear': Activation(linear),
        # Complex input, Real output
        'convert_to_real_with_abs': Activation(convert_to_real_with_abs),
        'softmax_real': Activation(softmax_real),
        # Cartesian form
        'cart_sigmoid': Activation(cart_sigmoid),
        'cart_elu': Activation(cart_elu),
        'cart_exponential': Activation(cart_exponential),
        'cart_hard_sigmoid': Activation(cart_hard_sigmoid),
        'cart_relu': Activation(cart_relu),
        'cart_leaky_relu': Activation(cart_leaky_relu),
        'cart_selu': Activation(cart_selu),
        'cart_softplus': Activation(cart_softplus),
        'cart_softsign': Activation(cart_softsign),
        'cart_tanh': Activation(cart_tanh),
        'cart_softmax': Activation(cart_softmax),
        # Polar form
        'pol_tanh': Activation(pol_tanh),
        'pol_sigmoid': Activation(pol_sigmoid),
        'pol_selu': Activation(pol_selu)
    }




.. note:: To define your own activation function, just create it inside the :code:`activations.py` and add it to :code:`act_dispatcher` dictionary at the end


.. py:method:: linear(z)

	Does not apply any activation function. It just outputs the input.
	
    :param z: Input tensor variable
    :return: z


.. toctree::
	:maxdepth: 2

    activations/types
    activations/real_output
    activations/mvn_activation.ipynb

