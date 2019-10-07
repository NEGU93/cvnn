Activation Functions
====================

This functions will be used when creating the graph of the network. For each corresponding string of options you can use any of the follwing activation functions.

Activation functions are created inside the class and referenced using the function::

	apply_activation(out, act)

.. note:: To add your own activation function just define the function in question and add it to the `apply_activation` method. 

.. note:: Convention: Method names have the following conventions. The predecind `act` means it is an activation function. The following `cart` or `polar` means either type A (cartesian) or type B (polar) according to [CIT2003-KUROE]_ notation.

.. py:class:: Cvnn

.. py:method:: apply_activation(self, out, act)
	Applies activation function to parameter out according to string act

        :param out: Tensor to whom the activation function will be applied
        :param act: string that says which activation function will be applied. If string does not correspond to any known activation function, none will be applied and a warning will be displayed.
        :return: Tensor with the applied activation function





.. [CIT2003-KUROE] Kuroe, Yasuaki, Mitsuo Yoshid, and Takehiro Mori. "On activation functions for complex-valued neural networks—existence of energy functions—." Artificial Neural Networks and Neural Information Processing—ICANN/ICONIP 2003. Springer, Berlin, Heidelberg, 2003. 985-992.
