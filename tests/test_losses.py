from cvnn.losses import ComplexAverageCrossEntropy, ComplexWeightedAverageCrossEntropy
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import cvnn.dataset as dp
from cvnn.layers import ComplexDense, complex_input
from pdb import set_trace


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


def weighted_loss():
    y_true = np.array([
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [0., 1.]
    ])
    y_pred = np.array([
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.]
    ])
    ace = ComplexAverageCrossEntropy()(y_pred=tf.complex(y_pred, y_pred), y_true=y_true)
    wace = ComplexWeightedAverageCrossEntropy(weights=[1., 9.])(y_pred=tf.complex(y_pred, y_pred), y_true=y_true)
    assert ace.numpy() < wace.numpy(), f"ACE {ace.numpy()} > WACE {wace.numpy()}"


def test_losses():
    weighted_loss()
    ace()


if __name__ == "__main__":
    test_losses()
