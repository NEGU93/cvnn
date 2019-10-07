Cvnn
===========

Upon construction, the object connects to the desired Cypress device.
For initializing the device there are 3 options according to the need.

.. python:cvnn:: Cvnn()

        Performs the training of the neural network
        :param x_train: Training data
        :param y_train: Labels of the training data
        :param x_test: Test data to display accuracy at the end
        :param y_test: Test labels
        :param epochs: Total number of training epochs
        :param batch_size: Training batch size
        :param display_freq: Display results frequency
        :return: None
