Train and Predict
=================

.. py:class:: Cvnn

.. py:method:: train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=100, display_freq=1000)

	Performs the training of the neural network. 
        If automatic_restore is True but not metadata was found, it will try to load the weights of the newest previously saved model.

        :param x_train: Training data of shape (<training examples>, <input_size>)
        :param y_train: Labels of the training data of shape (<training examples>, <output_size>)
        :param x_test: Test data to display accuracy at the end of shape (<test examples>, <input_size>)
        :param y_test: Test labels of shape (<test examples>, <output_size>)
        :param epochs: Total number of training epochs
        :param batch_size: Training batch size. If this number is bigger than the total amount of training examples will display an error
        :param display_freq: Display results frequency. The frequency will be for each (epoch * batch_size + iteration) / display_freq
        :return: None

.. py:method:: predict(self, x)

	Runs a single feedforward computation

        :param x: Input of the network
        :return: Output of the network
