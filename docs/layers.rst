Complex Layers
==============

.. py:class:: ComplexDense

Fully connected complex-valued layer
    Implements the operation:

    .. math::

        \sigma(\textrm{input * weights + bias}) 

    * where data types can be either complex or real.
    * activation (:math:`\sigma`) is the element-wise activation function passed as the activation argument, 
    * weights is a matrix created by the layer
    * bias is a bias vector created by the layer

.. py:method:: __init__(self, input_size, output_size, activation=None, input_dtype=np.complex64, output_dtype=np.complex64, weight_initializer=tf.keras.initializers.GlorotUniform, bias_initializer=tf.keras.initializers.Zeros)

        Initializer of the Dense layer

        :param input_size: Input size of the layer 
        :param output_size: Output size of the layer
        :param activation: Activation function to be used.
            Can be either the function from :ref:`activation_functions` or `tensorflow.python.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_
            or a string as listed in act_dispatcher
        :param input_dtype: data type of the input.

            Default: :code:`np.complex64`

            .. note::
                Supported data types:
                    * :code:`np.complex64`
                    * :code:`np.float32`

        :param output_dtype: data type of the output function. 
            Default: :code:`np.complex64`  
        :param weight_initializer: Initializer for the weights. 
            Default: `tensorflow.keras.initializers.GlorotUniform <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>`_
        :param bias_initializer: Initializer fot the bias. 
            Default: :code:`tensorflow.keras.initializers.Zeros`



