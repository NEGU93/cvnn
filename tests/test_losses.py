from cvnn.losses import ComplexAverageCrossEntropy, ComplexWeightedAverageCrossEntropy, \
    ComplexAverageCrossEntropyIgnoreUnlabeled
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from cvnn.layers import ComplexDense, complex_input
from pdb import set_trace


def to_categorical_unlabeled(sparse, classes=2):
    cat = np.zeros(shape=sparse.shape + (classes,))
    for i in range(len(sparse)):
        for row in range(len(sparse[i])):
            for col in range(len(sparse[i][row])):
                if sparse[i][row][col]:
                    cat[i][row][col][sparse[i][row][col] - 1] = 1
    return cat


def averaging_method():
    # Here, I see that the loss is not computed per image, but per pixel.
    y_true = np.array([
        [[1, 1], [1, 1]],
        [[0, 0], [0, 2]]
    ])
    y_pred = np.array([
        [[1, 1], [1, 2]],
        [[1, 1], [1, 1]]
    ])
    y_true = to_categorical_unlabeled(y_true)
    y_pred = to_categorical_unlabeled(y_pred)
    class_loss_result = CategoricalCrossentropy()(y_pred=y_pred, y_true=y_true)
    fun_loss_result = tf.keras.metrics.categorical_crossentropy(y_pred=y_pred, y_true=y_true)
    two_dim_mean = np.mean(fun_loss_result.numpy(), axis=(1, 2))
    mean = np.mean(fun_loss_result.numpy())
    assert np.allclose(mean, class_loss_result)
    assert np.mean(two_dim_mean) == mean


def no_label_test():
    y_true = np.array([
        [[0, 0], [0, 2]]
    ])
    y_true_2 = np.array([
        [[1, 2], [1, 2]]
    ])
    y_pred = np.array([
        [[1, 2], [1, 2]]
    ])
    y_true = to_categorical_unlabeled(y_true)
    y_pred = to_categorical_unlabeled(y_pred)
    y_true_2 = to_categorical_unlabeled(y_true_2)
    tf_result = ComplexAverageCrossEntropyIgnoreUnlabeled()(y_pred=tf.complex(y_pred, y_pred), y_true=y_true)
    tf_result_2 = ComplexAverageCrossEntropyIgnoreUnlabeled()(y_pred=tf.complex(y_pred, y_pred), y_true=y_true_2)
    assert tf_result == tf_result_2


def ace():
    y_pred = np.random.rand(3, 43, 12, 10)
    y_true = np.random.rand(3, 43, 12, 10)
    tf_result = CategoricalCrossentropy()(y_pred=y_pred, y_true=y_true)
    own_result = ComplexAverageCrossEntropy()(y_pred=tf.complex(y_pred, y_pred), y_true=y_true)
    own_real_result = ComplexAverageCrossEntropy()(y_pred=tf.convert_to_tensor(y_pred,  dtype=np.float64),
                                                   y_true=y_true)
    assert tf_result == own_real_result, f"ComplexCrossentropy {own_real_result} != CategoricalCrossentropy {tf_result}"
    assert tf_result == own_result, f"ComplexCrossentropy {own_result} != CategoricalCrossentropy {tf_result}"


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
    no_label_test()
    averaging_method()
    # weighted_loss()
    ace()


if __name__ == "__main__":
    test_losses()
