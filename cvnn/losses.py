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


if __name__ == "__main__":
    from cvnn.layers import ComplexDense, complex_input
    import cvnn.dataset as dp
    from pdb import set_trace
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    m = 10000
    n = 128
    param_list = [
        [0.3, 1, 1],
        [-0.3, 1, 1]
    ]
    dataset = dp.CorrelatedGaussianCoeffCorrel(m, n, param_list, debug=False)

    model = tf.keras.models.Sequential([
        complex_input(shape=(n)),
        ComplexDense(units=50, activation="cart_relu"),
        ComplexDense(2, activation="cart_softmax")
    ])

    model.compile(loss=ComplexAverageCrossEntropy(), metrics=["accuracy"], optimizer="sgd")

    model.fit(dataset.x, dataset.y, epochs=6)
