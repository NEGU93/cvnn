Cvnn
===========


.. py:function:: Cvnn(learning_rate=0.001, tensorboard=True, verbose=True, automatic_restore=True)

        Performs the training of the neural network

        :param x_train: Training data
        :param y_train: Labels of the training data
        :param x_test: Test data to display accuracy at the end
        :param y_test: Test labels
        :param epochs: Total number of training epochs
        :param batch_size: Training batch size
        :param display_freq: Display results frequency
        :return: None
