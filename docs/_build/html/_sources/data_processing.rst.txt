Data Processing
===============

Utils
-----

.. py:function:: randomize(x, y)

        Randomizes the order of data samples and their corresponding labels

        :param x: data
        :param y: data labels
        :return: Tuple of (shuffled_x, shuffled_y) maintaining coherence of elements labels


.. py:function:: get_next_batch(x, y, start, end)
	
	Get next batch from x and y using start and end

	:param x: data
	:param y: data labels
	:param start: starting index of the batch to be returned
	:param end: end index of the batch to be returned (not including)
	:return: tuple (x, y) of the selected batch

.. py:function:: transform_to_real(x_complex)

	:param x_complex: Complex-valued matrix of size mxn
	:return: real-valued matrix of size mx(2*n) unwrapping the real and imag part of the complex-valued input matrix

.. py:function:: separate_into_train_and_test(x, y, ratio=0.8)

	Separates data x with corresponding labels y into train and test set.

    	:param x: data
    	:param y: labels of data x
    	:param ratio: value between 0 and 1. 1 meaning all the data x will be the training set and 0 meaning all data x will be the test set.
    	:return: tuple (x_train, y_train, x_test, y_test) of the training and test set both data and labels.


Save and load dataset
---------------------

.. py:function:: save_dataset(array_name, x_train, y_train, x_test, y_test)

	Saves in a single .npz file the test and training set with corresponding labels

	:param array_name: Name of the array to be saved into data/ folder.
	:param x_train:
    	:param y_train:
    	:param x_test:
    	:param y_test:
    	:return: None


.. _load_dataset:
.. py:function:: load_dataset(array_name)

	Gets all x_train, y_train, x_test, y_test from a previously saved .npz file with :ref:`save_dataset` function.

	:param array_name: name of the file saved in '../data' with .npz termination
    	:return: tuple (x_train, y_train, x_test, y_test)

