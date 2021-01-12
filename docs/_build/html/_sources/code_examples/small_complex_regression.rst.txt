Small complex-regression example
--------------------------------

Let's just see how we can do a regression small CVNN.

As usual, first import what is needed:

.. code-block:: python

    import numpy as np
    import cvnn.layers as complex_layers
    import tensorflow as tf

Let's create random complex data

.. code-block:: python

    input_shape = (4, 28, 28, 3)
    x = tf.cast(tf.random.normal(input_shape), tf.complex64)

Now let's create our network and compile it

.. code-block:: python

    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=input_shape[1:]))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(units=64, activation='cart_relu'))
    model.add(complex_layers.ComplexDense(units=10, activation='linear'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


This is it! Now you can train uwing the :code:`fit` method or predict.
You can check for example that the output of the model is still complex (as expected).

.. code-block:: python

    y = model(x)
    assert y.dtype == np.complex64