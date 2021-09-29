import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, Precision, Recall
from tensorflow_addons.metrics import F1Score, CohenKappa


class ComplexAccuracy(Accuracy):

    def __init__(self, name='complex_accuracy', dtype=tf.complex64, **kwargs):
        super(ComplexAccuracy, self).__init__(name=name, dtype=dtype, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class ComplexCategoricalAccuracy(CategoricalAccuracy):

    def __init__(self, name='complex_categorical_accuracy', **kwargs):
        super(ComplexCategoricalAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexCategoricalAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class ComplexPrecision(Precision):

    def __init__(self, name='complex_precision', **kwargs):
        super(ComplexPrecision, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexPrecision, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class ComplexRecall(Recall):

    def __init__(self, name='complex_recall', **kwargs):
        super(ComplexRecall, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexRecall, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class ComplexCohenKappa(CohenKappa):

    def __init__(self, name='complex_cohen_kappa', **kwargs):
        super(ComplexCohenKappa, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexCohenKappa, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class ComplexF1Score(F1Score):

    def __init__(self, name='complex_f1_score', **kwargs):
        super(ComplexF1Score, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexF1Score, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


if __name__ == '__main__':
    m = ComplexAccuracy()
    m.update_state([[1+1j], [2+1j], [3+1j], [4+1j]], [[1+1j], [2+1j], [3+5j], [4+5j]])
    print(m.result().numpy())


