Losses
======

For the losses, if :code:`y_pred` is complex and :code:`y_true` is floating, :code:`y_true` will be automatically cast to complex by replicating its value on the imaginary part.

Complex Average Cross Entropy
-----------------------------

Inspired on [CIT2018-CAO]_ Average Cross Entropy (ACE) loss function described on section 2.4.

This function applies Categorical Cross Entropy to both the real and imaginary part separately and then averages it.
Mathematically this is

.. math::
 
    J^{ACE} = \frac{1}{2} \left[ J^{CCE}(\Re \hat{y}, y) + J^{CCE}(\Im \hat{y}, y) \right] \, ,

where :math:`J^{ACE}` is the Complex Average Cross Entropy, :math:`J^{CCE}` is the well known Categorical Cross Entropy. :math:`\hat{y}` is the predicted labels with the corresponding ground truth :math:`y`. Finally :math:`\Re` and :math:`\Im` operators are the real and imaginary parts of the input respectively.
For real-valued output :math:`J^{ACE} = J^{CCE}`.


Working example::

    
    from cvnn.layers import ComplexDense, complex_input
    from cvnn.losses import ComplexAverageCrossEntropy
    import cvnn.dataset as dp
    import tensorflow as tf

    # Get dataset
    m = 10000
    n = 128
    param_list = [
        [0.3, 1, 1],
        [-0.3, 1, 1]
    ]
    dataset = dp.CorrelatedGaussianCoeffCorrel(m, n, param_list, debug=False)

    # Build model
    model = tf.keras.models.Sequential([
        complex_input(shape=(n)),
        ComplexDense(units=50, activation="cart_relu"),
        ComplexDense(2, activation="cart_softmax")
    ])

    # Compile using ACE complex loss function
    model.compile(loss=ComplexAverageCrossEntropy(), metrics=["accuracy"], optimizer="sgd")

    model.fit(dataset.x, dataset.y, epochs=6)


Complex Mean Square Error
-------------------------

Performs the mean square error defined as:

.. math::
 
    \mathcal{L} =  \Delta x^{2} + \Delta y^{2} \, ,

where :math:`\Delta x` and :math:`\Delta y` represents the real and imaginary part difference between the label and predicted respectively.


Working example::

    import numpy as np
    import tensorflow as tf
    from cvnn.losses import ComplexMeanSquareError
    y_true = np.random.randint(0, 2, size=(2, 3)).astype("float32")
    y_pred = tf.complex(np.random.random(size=(2, 3)).astype("float32"),
                        np.random.random(size=(2, 3)).astype("float32"))
    loss = ComplexMeanSquareError().call(y_true, y_pred)
    expected_loss = np.mean(np.square(np.abs(tf.complex(y_true, y_true) - y_pred)), axis=-1)
    assert np.all(loss == expected_loss)


.. [CIT2018-CAO] Yice Cao, Yan Wu, Peng Zhang, Wenkai Liang and Ming Li "Pixel-Wise PolSAR Image Classification via a Novel Complex-Valued Deep Fully Convolutional Network" https://arxiv.org/abs/1909.13299 2019