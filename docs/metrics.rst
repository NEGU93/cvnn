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