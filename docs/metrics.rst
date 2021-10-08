Metrics
=======

The following metrics accept only real valued for :code:`y_true`.  


If :code:`y_pred` is real, it will converge to TensorFlow implementations.

If not, it will cast :code:`y_pred` to real by making :code:`y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2`.

Available metrics

- :code:`ComplexAccuracy`: Complex implementation of `Accuracy <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy>`_
- :code:`ComplexCategoricalAccuracy`: Complex implementation of `CategoricalAccuracy <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy>`_
- :code:`ComplexPrecision`: Complex implementation of `Precision <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision>`_
- :code:`ComplexRecall`: Complex implementation of `Recall <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Recall>`_
- :code:`ComplexCohenKappa`: Complex implementation of `CohenKappa <https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/CohenKappa>`_
- :code:`ComplexF1Score`: Complex implementation of `F1Score <https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score>`_

Complex Average Accuracy
------------------------

Average Accuracy (AA) is defined as the average of individual class accuracy. 
This is used for unbalanced dataset in order to see the actual accuracy per class.

For example::

    # Unbalanced dataset with 90% cases of one class and 10% of the other 
    y_true = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.] ])
    # Dummy classifier has learned to just predict always the first class
    y_pred = np.array([ [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.] ])
    m = ComplexCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](.9)     # The dummy classifier has a big accuracy of 90%
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](0.5)    # But an average accuracy of just 50%