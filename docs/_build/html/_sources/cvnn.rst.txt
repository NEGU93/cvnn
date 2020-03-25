CVNN
===========

.. py:class:: CvnnModel

.. py:method:: __init__(self, learning_rate=0.001, tensorboard=True, verbose=True, automatic_restore=True)

        Constructor

        :param name: Name of the model. It will be used to distinguish models
        :param shape: List of :code:`cvnn.layers.ComplexLayer` objects
        :param loss_fun: :code:`tensorflow.python.keras.losses` to be used.
        :param verbose: if :code:`True` it will print information of the model just created
        :param tensorboard: If :code:`True` it will save tensorboard information inside :code:`log/.../tensorboard_logs/`

                * Loss and accuracy
                * Graph
                * Weights histogram

        :param save_model_checkpoints: Save the model to be able to load and continue training later
        :param save_csv_checkpoints: Save information of the train and test loss and accuracy on csv files
        .. warning::
                :code:`save_model_checkpoints` Not yet working.
        

Train
-----

.. py:method:: fit(self, x, y, ratio=0.8, learning_rate=0.01, epochs=10, batch_size=32,
            verbose=True, display_freq=None, fast_mode=False, save_to_file=True)

	Trains the model for a fixed number of epochs (iterations on a dataset).

        :param x: Input data. 
        :param y: Labels
        :param ratio: Percentage of the input data to be used as train set (the rest will be use as validation set)
            Default: 0.8 (80% as train set and 20% as validation set)
        :param learning_rate: Learning rate for the gradient descent. For the moment only GD is supported.
        :param epochs: (uint) Number of epochs to do.
        :param batch_size: (uint) Batch size of the data. Default 32 (because keras use 32 so... why not?)
        :param verbose: (Boolean) Print results of the training while training
        :param display_freq: Frequency on terms of steps for saving information and running a checkpoint.
            If :code:`None` (default) it will automatically match 1 epoch = 1 step (print/save information at each epoch)
        :param fast_mode: (Boolean) Takes precedence over :code:`verbose` and :code:`save_to_file`
        :param save_to_file: (Boolean) save a txt with the information of the fit
                    (same as what will be printed if :code:`verbose`)
        :return: None

Results
-------

.. py:method:: call(self, x)

        Forward result of the network

        :param x: Data input to be calculated
        :return: Output of the netowrk

.. py:method:: predict(self, x)

	Predicts the value of the class.
        
        .. warning:: 
                ATTENTION: Use this only for classification tasks. For regression use :code:`call` method.

        :param x: Input
        :return: Prediction of the class that x belongs to.

.. py:method:: evaluate_loss(self, x, y)

	Computes the output of x and computes the loss using y

        :param x: Input of the netwotk
        :param y: Labels
        :return: loss value

.. py:method:: evaluate_accuracy(self, x, y)

        Computes the output of x and returns the accuracy using y as labels

        :param x: Input of the netwotk
        :param y: Labels
        :return: accuracy

.. py:method:: evaluate(self, x, y)

        Compues both the loss and accuracy using :code:`evaluate_loss` and :code:`evaluate_accuracy`

        :param x: Input of the netwotk
        :param y: Labels
        :return: tuple (loss, accuracy)

Others
------

.. py:method:: summary(self)

	Generates a string of a summary representation of your model.

        :return: string of the summary of the model

.. py:method:: is_complex(self)

        :return: :code:`True` if the network is complex. :code:`False` otherwise.
