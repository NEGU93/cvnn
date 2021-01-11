Complex Dropout
---------------

.. py:class:: ComplexDropout

    Computes dropout: randomly sets elements to zero to prevent overfitting.

    Dropout [CIT2014-SRIVASTAVA]_ consists in randomly setting a fraction :code:`rate` of input units to 0 at each update during training time, which helps prevent overfitting.

.. py:method:: __init__(self, rate, noise_shape=None, seed=None)
        
        :param rate: A scalar Tensor with the same type as x.
            The probability that each element is dropped.
            For example, setting :code:`rate=0.1` would drop 10% of input elements.
        :param noise_shape: A 1-D Tensor of type :code:`int32`, representing the shape for randomly generated keep/drop flags.
        :param seed:  A Python integer. Used to create random seeds. See :code:`tf.random.set_seed` for behavior.


.. [CIT2014-SRIVASTAVA] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, Jan. 2014