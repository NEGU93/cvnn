CVNN
===========

Short example::

        # Assume you already have complex data 'x' with its labels 'y'...

        shape = [layers.ComplexDense(output_size=100, input_size=np.shape(x)[1], activation='cart_relu'),
                layers.ComplexDense(output_size=40, activation='cart_relu'),
                layers.ComplexDense(output_size=np.shape(y)[1], activation='softmax_real')]
        model = CvnnModel("cvnn_example", shape, tf.keras.losses.categorical_crossentropy)
        model.fit(x, y, batch_size=100, epochs=150)


.. py:class:: CvnnModel


.. py:method:: __init__(self, name, shape, loss_fun, optimizer='sgd', verbose=True, tensorboard=True)

        Constructor

        :param name: Name of the model. It will be used to distinguish models
        :param shape: List of :code:`cvnn.layers.ComplexLayer` objects
        :param loss_fun: :code:`tensorflow.python.keras.losses` to be used.
        :param optimizer: Optimizer to be used. Keras optimizers are not allowed. Can be either :code:`cvnn.optimizers.Optimizer` or a string listed in :code:`opt_dispatcher`.
        :param verbose: if :code:`True` it will print information of the model just created
        :param tensorboard: If :code:`True` it will save tensorboard information inside :code:`log/.../tensorboard_logs/`

                * Loss and accuracy
                * Graph
                * Weights histogram

Train
-----

.. py:method:: fit(x, y=None, validation_split=0.0, validation_data=None, epochs: int = 10, batch_size: int = 32, verbose=True, display_freq: int = 1, save_model_checkpoints=False, save_csv_history=True, shuffle=True)

	Trains the model for a fixed number of epochs (iterations on a dataset).

        :param x: Input data. It could be:
            - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            - A :code:`TensorFlow tensor`, or a list of tensors (in case the model has multiple inputs).
            - A :code:`tf.data dataset`. Should return a tuple (inputs, targets). Preferred data type (less overhead).
        :param y: Labels/Target data. Like the input data :code:`x`, it could be either Numpy array(s) or TensorFlow tensor(s).
            If :code:`x` is a dataset then y will be ignored (default :code:`None`)
        :param validation_split: Float between 0 and 1.
            Percentage of the input data to be used as test set (the rest will be use as train set)
            Default: 0.0 (No validation set).
            This input is ignored if :code:`validation_data` is given.
        :param validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. This parameter takes precedence over :code:`validation_split`.
            It can be:
                - tuple :code:`(x_val, y_val)` of Numpy arrays or tensors. Preferred data type (less overhead).
                - A :code:`tf.data dataset`.
        :param epochs: (:code:`uint`) Number of epochs to do.
        :param batch_size: (:code:`uint`) Batch size of the data. Default :code:`32` (because keras use 32 so... why not?)
        :param verbose: Verbosity Mode
            It can be:
                - Bool: False defaults to 0 and True to 1.
                - Int
                - String: Matching the modes string
            Verbosity Modes:
                - "SILENT" or 0:  No prints of any kind
                - "FAST" or 2:    Does not show the progress bar of each epoch.
                    Verbosity modes "FAST" and "SILENT" saves the csv file (if save_csv_history) less often.
                    Making it faster riskier of data loss
                - "PROBAR" or 4:  Shows progress bar but does not show accuracy or loss (helps on speed)
                - "INFO" or 1:    Shows a progress bar with current accuracy and loss
                - "DEBUG" or 3:   Shows start and end messages and also the progress bar with current accuracy and loss
            Verbosity modes 0, 1 and 2 are coincident with `tensorflow's fit verbose parameter <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
        :param display_freq: Integer (Default 1)
            Frequency on terms of epochs before saving information and running a checkpoint.
        :param save_model_checkpoints: (Boolean)
                    Save the model to be able to load and continue training later
        :param save_csv_history: (Boolean) Save information of the train and test loss and accuracy on csv files.
        :param shuffle: (Boolean) Whether to shuffle the training data before each epoch. Default: True
        :return: None

        .. warning::
                :code:`save_model_checkpoints` Not yet working. So default is False and will through error otherwise.
        

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

.. py:method:: get_confusion_matrix(self, x, y, save_result=False)

        Generates a pandas data-frame with the confusion matrix of result of x and y (labels)

        :param x: data to which apply the model
        :param y: labels
        :param save_result: if :code:`True` it will save the confusion matrix as a csv at models path
        :return: Confusion matrix pandas data-frame

Others
------

.. py:method:: summary(self)

	Generates a string of a summary representation of your model::

                model.summary()
                

        :return: string of the summary of the model

.. py:method:: is_complex(self)

        :return: :code:`True` if the network is complex. :code:`False` otherwise::

                # x dtype is np.complex64
                if not model.is_complex():
                        x = cvnn.utils.transform_to_real(x)

.. py:method:: get_real_equivalent(self, classifier=True, name=None)
        
        Creates a new model equivalent of current model. If model is already real throws and error.

        :param classifier: :code:`True` (default) if the model is a classification model. :code:`False` otherwise.
        :param name: name of the new network to be created.
            If :code:`None` (Default) it will use same name as current model with "_real_equiv" suffix
        :return: :code:`CvnnModel()` real equivalent model

