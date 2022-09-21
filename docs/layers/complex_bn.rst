Complex Batch Normalization
---------------------------

.. py:class:: ComplexBatchNormalization

    Complex Batch-Normalization as defined in section 3.5 of [TRABELESI-2017]_

.. py:method:: __init__(axis=-1, momentum=0.99, center=True, scale=True, epsilon=0.001, beta_initializer=Zeros(), gamma_initializer=Ones(), dtype=DEFAULT_COMPLEX_TYPE, moving_mean_initializer=Zeros(), moving_variance_initializer=Ones(), cov_method: int = 2,  **kwargs)

    :param axis: Integer, the axis that should be normalized (typically the features axis). 
    :param momentum: Float. Momentum for the moving average.
    :param center: If True, add offset of beta to normalized tensor. If False, beta is ignored.
    :param scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.
    :param epsilon: Small float added to variance to avoid dividing by zero. 
    :param beta_initializer: Initializer for the beta weight.
    :param gamma_initializer: Initializer for the gamma weight.
    :param dtype: tf.complex32
    :param moving_mean_initializer: Initializer for the moving mean.
    :param moving_variance_initializer: Initializer for the moving variance.
    :param cov_method: Either 1 or 2. Algorithm to be applie. It should be the same results but calculated in a different manner. Used for debugging.


.. [TRABELESI-2017] Trabelsi, Chiheb et al. "Deep Complex Networks" arXiv:1705.09792 [cs]. 2017.