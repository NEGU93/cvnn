Complex Metrics
===============

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

.. py:method:: update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True)

    :param y_true: Ground truth label values. 
    :param y_pred: The predicted probability values.
    :param sample_weight: Optional :code:`sample_weight` acts as a coefficient for the metric. 
        If a scalar is provided, then the metric is simply scaled by the given value. 
        If :code:`sample_weight` is a tensor of size :code:`[batch_size]`, then the metric for each sample of the batch is rescaled by the corresponding element in the :code:`sample_weight` vector. 
        If the shape of :code:`sample_weight` is :code:`[batch_size, d0, .. dN-1]`` (or can be broadcasted to this shape), then each metric element of y_pred is scaled by the corresponding value of :code:`sample_weight`. 
        (Note on dN-1: all metric functions reduce by 1 dimension, usually the last axis :code:`(-1)`).
    :param ignore_unlabeled: Default :code:`True`. Ignore cases where :code:`labels[-1] == zeros`.
        The :code:`sample_weight` parameter is used to ignore unlabeled data so using this will deprect the :code:`sample_weight` parameter.

.. warning::

    :code:`ignore_unlabeled` takes precedence over :code:`sample_weight` so make sure to turn it to :code:`False` when using :code:`sample_weight`


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
    print(m.result().numpy())     # The dummy classifier has a big accuracy of 90%
        >>> 0.9
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    print(m.result().numpy())     # But an average accuracy of just 50%
        >>> 0.5