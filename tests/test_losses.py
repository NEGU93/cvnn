from cvnn.losses import ComplexAverageCrossEntropy
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import cvnn.dataset as dp
from cvnn.layers import ComplexDense, complex_input


def ace():
    y_pred = np.random.rand(3, 43, 12, 10)
    y_true = np.random.rand(3, 43, 12, 10)
    tf_result = CategoricalCrossentropy()(y_pred=y_pred, y_true=y_true)
    own_result = ComplexAverageCrossEntropy()(y_pred=tf.complex(y_pred, y_pred), y_true=y_true)
    assert tf_result == own_result, f"ComplexCrossentropy {own_result} != CategoricalCrossentropy {tf_result}"
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


def test_losses():
    ace()


if __name__ == "__main__":
    test_losses()
