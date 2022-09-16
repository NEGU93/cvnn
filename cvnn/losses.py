import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.losses import Loss, categorical_crossentropy


class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):
        real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
        if y_pred.dtype.is_complex:
            imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2.


class ComplexAverageCrossEntropyIgnoreUnlabeled(ComplexAverageCrossEntropy):

    def call(self, y_true, y_pred):
        mask = tf.reduce_any(tf.cast(y_true, tf.bool), axis=-1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return super(ComplexAverageCrossEntropyIgnoreUnlabeled, self).call(y_true, y_pred)


class ComplexMeanSquareError(Loss):

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        if y_pred.dtype.is_complex and not y_true.dtype.is_complex:     # Complex pred but real true
            y_true = tf.complex(y_true, y_true)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.cast(backend.mean(tf.math.square(tf.math.abs(y_true - y_pred)), axis=-1),
                       dtype=y_pred.dtype.real_dtype)


class ComplexWeightedAverageCrossEntropy(ComplexAverageCrossEntropy):

    def __init__(self, weights, **kwargs):
        self.class_weights = weights
        super(ComplexWeightedAverageCrossEntropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        # https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        unweighted_losses = super(ComplexWeightedAverageCrossEntropy, self).call(y_true, y_pred)
        weighted_losses = unweighted_losses * tf.cast(weights, dtype=unweighted_losses.dtype)
        return weighted_losses


class ComplexWeightedAverageCrossEntropyIgnoreUnlabeled(ComplexAverageCrossEntropy):

    def __init__(self, weights, **kwargs):
        self.class_weights = weights
        super(ComplexWeightedAverageCrossEntropyIgnoreUnlabeled, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        mask = tf.reduce_any(tf.cast(y_true, tf.bool), axis=-1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return super(ComplexWeightedAverageCrossEntropyIgnoreUnlabeled, self).call(y_true, y_pred)


if __name__ == "__main__":
    import numpy as np
    y_true = np.random.randint(0, 2, size=(2, 3)).astype("float32")
    y_pred = tf.complex(np.random.random(size=(2, 3)).astype("float32"),
                        np.random.random(size=(2, 3)).astype("float32"))
    loss = ComplexMeanSquareError().call(y_true, y_pred)
    expected_loss = np.mean(np.square(np.abs(tf.complex(y_true, y_true) - y_pred)), axis=-1)
    # import pdb; pdb.set_trace()
    assert np.all(loss == expected_loss)
