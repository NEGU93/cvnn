import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, Precision, Recall, Mean
from tensorflow_addons.metrics import F1Score, CohenKappa
from tensorflow.python.keras import backend


class ComplexAccuracy(Accuracy):

    def __init__(self, name='complex_accuracy', dtype=tf.complex64, **kwargs):
        super(ComplexAccuracy, self).__init__(name=name, dtype=dtype, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        if ignore_unlabeled:    # WARNING, this will overwrite sample_weight!
            sample_weight = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
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

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        if ignore_unlabeled:    # WARNING, this will overwrite sample_weight!
            sample_weight = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
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

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        if ignore_unlabeled:    # WARNING, this will overwrite sample_weight!
            sample_weight = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
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

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        if ignore_unlabeled:    # WARNING, this will overwrite sample_weight!
            sample_weight = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
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

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        if ignore_unlabeled:    # WARNING, this will overwrite sample_weight!
            sample_weight = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
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

    def update_state(self, y_true, y_pred, sample_weight=None, ignore_unlabeled=True):
        if ignore_unlabeled:    # WARNING, this will overwrite sample_weight!
            sample_weight = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:
            assert tf.math.reduce_all(tf.math.real(y_pred) == tf.math.imag(y_pred)), "y_pred must be real valued"
        super(ComplexF1Score, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


def _accuracy(y_true, y_pred):
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    reduced_sum = tf.reduce_sum(tf.cast(tf.math.equal(y_true, y_pred), backend.floatx()), axis=-1)
    return tf.math.divide_no_nan(reduced_sum, tf.cast(tf.shape(y_pred)[-1], reduced_sum.dtype))


def custom_average_accuracy(y_true, y_pred):
    # Mask to remove the labels (y_true) that are zero: ex. [0, 0, 0]
    remove_zeros_mask = tf.math.logical_not(tf.math.reduce_all(tf.math.logical_not(tf.cast(y_true, bool)), axis=-1))
    y_true = tf.boolean_mask(y_true, remove_zeros_mask)
    y_pred = tf.boolean_mask(y_pred, remove_zeros_mask)
    num_cls = y_true.shape[-1]                  # get total amount of classes
    y_pred = tf.math.argmax(y_pred, axis=-1)    # one hot encoded to sparse
    y_true = tf.math.argmax(y_true, axis=-1)    # ex. [0, 0, 1] -> [2]
    accuracies = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in range(0, num_cls):
        cls_mask = y_true == i
        cls_y_true = tf.boolean_mask(y_true, cls_mask)
        if tf.not_equal(tf.size(cls_y_true), 0):
            new_acc = _accuracy(y_true=cls_y_true, y_pred=tf.boolean_mask(y_pred, cls_mask))
            accuracies = accuracies.write(accuracies.size(), new_acc)
    # import pdb; pdb.set_trace()
    accuracies = accuracies.stack()
    # return tf.cast(len(accuracies), dtype=accuracies.dtype)
    return tf.math.reduce_sum(accuracies) / tf.cast(len(accuracies), dtype=accuracies.dtype)


class ComplexAverageAccuracy(Mean):

    def __init__(self, name='complex_average_accuracy', dtype=None):
        self._fn = custom_average_accuracy
        super(ComplexAverageAccuracy, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # WARNING: sample_weights will not be used
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex:     # make y_pred real valued
            y_pred = (tf.math.real(y_pred) + tf.math.imag(y_pred)) / 2
        if y_true.dtype.is_complex:     # make y_true real valued
            assert tf.math.reduce_all(tf.math.real(y_true) == tf.math.imag(y_true)), "y_pred must be real valued"
            y_true = tf.math.real(y_true)
        matches = self._fn(y_true, y_pred)
        return super(ComplexAverageAccuracy, self).update_state(matches)


if __name__ == '__main__':
    m = ComplexAccuracy()
    m.update_state([[1+1j], [2+1j], [3+1j], [4+1j]], [[1+1j], [2+1j], [3+5j], [4+5j]])
    print(m.result().numpy())


