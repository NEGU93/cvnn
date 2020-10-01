Optimizers
==========

.. py:class:: Optimizer(ABC)

    All optimizers defined here inherit from :code:`Optimizer`

.. py:method:: summary()

    :returns: A one line short string with the description of the optimizer



Stochastic Gradiend Descent (SGD)
---------------------------------

.. py:class:: SGD

    Gradient descent (with momentum) optimizer.

.. py:method:: __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, name: str = 'SGD')

    :param learning_rate: The learning rate. Defaults to 0.001.
    :param momentum: float hyperparameter between [0, 1) that accelerates gradient descent in the relevant direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
    :param name: Optional name for the operations created when applying gradients. Defaults to "Adam".

RMSprop
-------

.. py:class:: RMSprop

    Optimizer that implements the RMSprop algorithm. 
    `Reference <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_
    
    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average
    - This implementation of RMSprop uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the gradients, and uses that average to estimate the variance.

.. py:method:: __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, name="RMSprop")

    :param learning_rate: The learning rate. Defaults to 0.001.
    :param rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
    :param momentum: The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    :param epsilon: A small constant for numerical stability. Default 1e-07.
    :param name: Optional name for the operations created when applying gradients. Defaults to "Adam".


Adam (ADAptive Moment estimation)
---------------------------------

.. warning::

    Adam implementation appears to be rendering lower results than `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_ implementation.
    Further debugging is required.

.. py:class:: Adam

    Optimizer that implements the Adam algorithm.

    `Reference <https://arxiv.org/abs/1412.6980>`_: [KINGMA2015]_

    Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

.. py:method:: __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-07, name="Adam")

    :param learning_rate: The learning rate. Defaults to 0.001.
    :param beta_1: The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    :param beta_2: The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    :param epsilon: A small constant for numerical stability. Default 1e-07.
    :param name: Optional name for the operations created when applying gradients. Defaults to "Adam".



.. [KINGMA2015] Diederik P. Kingma, Jimmy Ba "Adam: A Method for Stochastic Optimization" arXiv:1412.6980 LG cs, 2015. Available: https://arxiv.org/abs/1412.6980