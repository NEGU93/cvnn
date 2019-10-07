Cvnn
===========

.. py:class:: Cvnn

.. py:method:: __init__(self, learning_rate=0.001, tensorboard=True, verbose=True, automatic_restore=True)

        Constructor

        :param learning_rate: Learning rate at which the network will train
        :param tensorboard: True if want the network to save tensorboard graph and summary
        :param verbose: True for verbose mode (print and output results)
        :param automatic_restore: True if network should search for saved models (will look for the newest saved model)


.. py:method:: __del__(self)

	Destructor


Train and Predict
-----------------

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

Graph Creation
--------------

Graphs
^^^^^^

.. py:method:: create_mlp_graph(self, shape)

	Creates a complex-fully-connected dense graph using a shape as parameter

        :param shape: List of tuple
            1. each number of shape[i][0] correspond to the total neurons of layer i.
            2. a string in shape[i][1] corresponds to the activation function listed on :ref:`activation_functions`
                ATTENTION: shape[0][0] will be ignored! A future version will apply the activation function to the input but not implemented for the moment.
            Where i = 0 corresponds to the input layer and the last value of the list corresponds to the output layer.
        :return: None

Others
^^^^^^

.. py:method:: restore_graph_from_meta(self, latest_file=None)
	
	Restores an existing graph from meta data file

        :param latest_file: Path to the file to be restored. If no latest_file given and self.automatic_restore is True, the function will try to load the newest metadata inside `saved_models/` folder.
        :return: None
