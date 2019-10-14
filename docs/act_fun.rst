.. _activation_functions:
Activation Functions
====================

This functions will be used when creating the graph of the network. For each corresponding string of options you can use any of the follwing activation functions.

Activation functions are created inside the class and referenced using the function::

	apply_activation(out, act)

.. note:: You can define your own activation function, just create it inside the Cvnn class. The function must however be static.

.. note:: Convention: The predecind `act` on `Cvnn` methods means it is an activation function. The following `cart` or `polar` means either type A (cartesian) or type B (polar) according to [CIT2003-KUROE]_ notation.

.. py:class:: Cvnn

.. py:method:: apply_activation(self, act, out)

	Applies activation function `act` to variable `out`
        :param out: Tensor to whom the activation function will be applied
        :param act: function to be applied to out. See the list fo possible activation functions on: `Implemented Activation Functions`_
            
        :return: Tensor with the applied activation function


Implemented Activation Functions
--------------------------------

.. py:method:: act_null(z)

	Does not apply any activation function. It just outputs the input

	:param z: Input tensor variable
        :return: z

.. py:method:: act_cart_sigmoid(z)

	Called with `'act_cart_sigmoid'` string. 
	Applies the function 

	.. math::

		\frac{1}{1 + e^{-x}} + j  \frac{1}{1 + e^{-y}}

	where 

	.. math::

		z = x + j y

        :param z: Tensor to be used as input of the activation function
        :return: Tensor result of the applied activation function






.. [CIT2003-KUROE] Kuroe, Yasuaki, Mitsuo Yoshid, and Takehiro Mori. "On activation functions for complex-valued neural networks—existence of energy functions—." Artificial Neural Networks and Neural Information Processing—ICANN/ICONIP 2003. Springer, Berlin, Heidelberg, 2003. 985-992.
