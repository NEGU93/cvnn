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


.. _transform-to-real-label:
.. py:function:: transform_to_real(x_complex, mode="real_imag")

	Transforms a complex input matrix into a real value matrix (double size)

    :param x_complex: Complex-valued matrix of size mxn
    :param mode: Mode on how to transform to real. One of the following.

        - :code:`real_imag` (default): Separate x_complex into real and imaginary making the size of the return double :code:`x_complex`
        - :code:`amplitude_phase`: Separate :code:`x_complex` into amplitude and phase making the size of the return double :code:`x_complex`
        - :code:`amplitude_only`: Apply the absolute value to :code:`x_complex`. Shape remains the same.
    :return: real-valued matrix of real valued cast of :code:`x_complex`


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