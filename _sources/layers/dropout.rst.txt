Complex Dropout
---------------

.. py:class:: ComplexDropout

    Applies Dropout to the input. It works also with complex inputs!
    The Dropout layer randomly sets input units to 0 with a frequency of :code:`rate` at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by :code:`1/(1 - rate)` such that the sum over all inputs is unchanged.
    Note that the Dropout layer only applies when :code:`training` is set to True such that no values are dropped during inference. 
    When using :code:`model.fit`, :code:`training` will be appropriately set to True automatically, and in other contexts, you can set the :code:`kwarg` explicitly to :code:`True` when calling the layer.
    This is in contrast to setting :code:`trainable=False` for a Dropout layer. :code:`trainable` does not affect the layer's behavior, as Dropout does not have any variables/weights that can be frozen during training.

    Dropout [CIT2014-SRIVASTAVA]_ consists in randomly setting a fraction :code:`rate` of input units to 0 at each update during training time, which helps prevent overfitting.

.. py:method:: __init__(self, rate, noise_shape=None, seed=None)
        
    :param rate: Float between 0 and 1. Fraction of the input units to drop.
    :param noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input.
        For instance, if your inputs have shape :code:`(batch_size, timesteps, features)` and you want the dropout mask to be the same for all timesteps, you can use :code:`noise_shape=(batch_size, 1, features)`.
    :param seed: A Python integer to use as random seed.

.. py:method:: call(self, inputs, training=None)

    :param inputs: Input tensor (of any rank).
    :param training: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing).

**Code example**

Let's first create some data:

.. code-block:: python

    import tensorflow as tf
    import numpy as np
    import cvnn.layers as complex_layers

    tf.random.set_seed(0)
    layer = complex_layers.ComplexDropout(.2, input_shape=(2,))
    data = np.arange(10).reshape(5, 2).astype(np.float32)
    data = tf.complex(data, data)
    print(data)

Data will therefore be::

    <tf.Tensor: shape=(5, 2), dtype=complex64, numpy=
    array([[0.+0.j, 1.+1.j],
          [2.+2.j, 3.+3.j],
          [4.+4.j, 5.+5.j],
          [6.+6.j, 7.+7.j],
          [8.+8.j, 9.+9.j]], dtype=complex64)>


Now when we apply the dropout layer:

.. code-block:: python

    outputs = layer(data, training=True)
    print(output)


It outputs:

.. code-block:: python

    <tf.Tensor: shape=(5, 2), dtype=complex64, numpy=
    array([[ 0.   +0.j  ,  1.25 +1.25j],
          [ 2.5  +2.5j ,  3.75 +3.75j],
          [ 5.   +5.j  ,  6.25 +6.25j],
          [ 7.5  +7.5j ,  8.75 +8.75j],
          [10.  +10.j  ,  0.   +0.j  ]], dtype=complex64)>

However, if you use :code:`training=False`, you will get the data unchanged::

    assert np.all(data == layer(data, training=False))

.. [CIT2014-SRIVASTAVA] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, Jan. 2014