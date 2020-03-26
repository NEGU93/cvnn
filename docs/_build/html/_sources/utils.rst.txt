Utils
=====


.. py:function:: get_func_name(fun)

	Returns the name of a function passed as parameter being either a function itself or a string with the function name::

		from cvnn.utils import get_func_name

		get_func_name(print)	# output: 'print'
		get_func_name('print')	# output: 'print'

    :param fun: function or function name
    :return: function name

.. py:function:: create_folder(root_path, now=None)
    
    Creates folders within :code:`root_path` using a date format.

    :param root_path: root path where to create the folder chain
    :param now: date to be used. If :code:`None` then it will use current time
    :return: the created path in pathlib format (compatible across different OS)


.. py:function:: transform_to_real(x_complex, polar=False)

	Transforms a complex input matrix into a real value matrix (double size)

    :param x_complex: Complex-valued matrix of size mxn
    :param polar: If :code:`True`, the data returned will be the amplitude and phase instead of real an imaginary part
        (Default: :code:`False`)
    :return: real-valued matrix of size mx(2*n) unwrapping the real and imag part of the complex-valued input matrix


.. py:function:: randomize(x, y):
    
    Randomizes the order of data samples and their corresponding labels

    :param x: data
    :param y: data labels
    :return: Tuple of (shuffled_x, shuffled_y) maintaining coherence of elements labels

.. py:function:: polar2cart(rho, angle):

    .. math::

		z = \rho \cdot e^{j\phi}

    :param rho: absolute value 
    :param angle: phase
    :return: complex number using phase and angle

.. py:function:: cart2polar(z):
    
    :param z: complex input
    :return: tuple with the absolute value of the input and the phase