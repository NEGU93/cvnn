Complex Dense
-------------

.. py:class:: ComplexDense

    Fully connected complex-valued layer

    Implements the operation:

    .. math::

        \sigma(\textrm{input * weights + bias}) 

    * where data types can be either complex or real.
    * activation (:math:`\sigma`) is the element-wise activation function passed as the activation argument, 
    * weights is a matrix created by the layer
    * bias is a bias vector created by the layer
    * dropout can be applied as well after the activation function if specified

.. py:method:: __init__(self, output_size, input_size=None, activation=None, input_dtype=None, weight_initializer=tf.keras.initializers.GlorotUniform, bias_initializer=tf.keras.initializers.Zeros, dropout=None)

        Initializer of the Dense layer

        :param output_size: Output size of the layer
        :param input_size: Input size of the layer (if :code:`None` it will automatically be defined)
        :param activation: Activation function to be used.
            Can be either the function from :ref:`activation_functions` or `tensorflow.python.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_
            or a string as listed in :code:`act_dispatcher`
        :param input_dtype: data type of the input. If :code:`None` (default) it will be defined automatically. 
        :param weight_initializer: Initializer for the weights. 
            Default: `tensorflow.keras.initializers.GlorotUniform <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>`_
        :param bias_initializer: Initializer fot the bias. 
            Default: :code:`tensorflow.keras.initializers.Zeros`
        :param dropout: Either None (default) and no dropout will be applied or a scalar that will be the probability that each element is dropped.

            Example: setting rate=0.1 would drop 10% of input elements.

.. py:method:: get_real_equivalent(self, output_multiplier=2)
        
        :param output_multiplier: Multiplier of output and input size (normally by 2). Can be used 1 for the output layer of a classifier.
        :return: real-valued copy of self


.. py:class:: ComplexDropout

    Computes dropout: randomly sets elements to zero to prevent overfitting.

    Dropout [CIT2014-SRIVASTAVA]_ consists in randomly setting a fraction :code:`rate` of input units to 0 at each update during training time, which helps prevent overfitting.

.. py:method:: __init__(self, rate, noise_shape=None, seed=None)
        
        :param rate: A scalar Tensor with the same type as x.
            The probability that each element is dropped.
            For example, setting rate=0.1 would drop 10% of input elements.
        :param noise_shape: A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
        :param seed:  A Python integer. Used to create random seeds. See tf.random.set_seed for behavior.


.. [CIT2014-SRIVASTAVA] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, Jan. 2014