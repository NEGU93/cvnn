import tensorflow as tf
from tensorflow.keras.losses import Loss, categorical_crossentropy


class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):
        real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
        if y_pred.dtype.is_complex:
            imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2.


class ComplexWeightedAverageCrossEntropy(ComplexAverageCrossEntropy):

    def __init__(self, weights, **kwargs):
        self.class_weights = weights
        super(ComplexWeightedAverageCrossEntropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        # https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        unweighted_losses = super(ComplexWeightedAverageCrossEntropy, self).call(y_true, y_pred)
        # import pdb; pdb.set_trace()
        weighted_losses = unweighted_losses * weights
        return weighted_losses
