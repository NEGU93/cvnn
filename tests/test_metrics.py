import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow as tf
from pdb import set_trace
from cvnn.metrics import ComplexAverageAccuracy, ComplexCategoricalAccuracy


def test_with_tf():
    classes = 3
    y_true = tf.cast(tf.random.uniform(shape=(34, 54, 12), maxval=classes), dtype=tf.int32)
    y_pred = tf.cast(tf.random.uniform(shape=y_true.shape, maxval=classes), dtype=tf.int32)
    y_pred_one_hot = tf.one_hot(y_pred, depth=classes)
    y_true_one_hot = tf.one_hot(y_true, depth=classes)
    tf_metric = CategoricalAccuracy()
    tf_metric.update_state(y_true_one_hot, y_pred_one_hot)
    own_metric = ComplexCategoricalAccuracy()
    own_metric.update_state(y_true_one_hot, y_pred_one_hot)
    # set_trace()
    assert own_metric.result().numpy() == tf_metric.result().numpy()
    y_true = np.array([
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [0., 0., 0., 0.],  # This shows tf does not ignore cases with [0. 0. 0. 0.] (unlabeled)
        [0., 0., 1., 0.],  # 3
        [0., 0., 1., 0.],  # 3
        [0., 0., 0., 0.],  # 3
        [0., 0., 1., 0.]  # 3
    ])
    y_pred = np.array([
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [0., 0., 0., 1.],  # 4
        [0., 0., 0., 1.],  # 4
        [0., 0., 0., 1.],  # 4
        [0., 0., 0., 1.]  # 4
    ])
    tf_metric = CategoricalAccuracy()
    tf_metric.update_state(y_true, y_pred)
    own_metric = ComplexCategoricalAccuracy()
    own_metric.update_state(y_true, y_pred, ignore_unlabeled=False)     # to make it as tf
    assert own_metric.result().numpy() == tf_metric.result().numpy()
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
    tf_metric = CategoricalAccuracy()
    tf_metric.update_state(y_true, y_pred)
    own_metric = ComplexCategoricalAccuracy()
    own_metric.update_state(y_true, y_pred, ignore_unlabeled=False)  # to make it as tf
    assert own_metric.result().numpy() == tf_metric.result().numpy()


def test_metric():
    y_true = [[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0], [0, 1, 0],
              [1, 0, 0]]
    y_pred = [[0.1, 0.9, 0.8],
              [0.1, 0.9, 0.8],
              [0.05, 0.95, 0], [0.95, 0.05, 0],
              [0, 1, 0]]

    m = ComplexCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == 0.25
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](1/6)     # I want 0.5/3 = 1/6
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
    m = ComplexCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](.9)
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](0.5)


def test_null_label():
    y_true = np.array([
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [0., 0., 1., 0.],  # 3
        [0., 0., 1., 0.],  # 3
        [0., 0., 1., 0.],  # 3
        [0., 0., 1., 0.]  # 3
    ])
    y_pred = np.array([
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [1., 0., 0., 0.],  # 1
        [0., 0., 0., 1.],  # 4
        [0., 0., 0., 1.],  # 4
        [0., 0., 0., 1.],  # 4
        [0., 0., 0., 1.]  # 4
    ])
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](0.5)
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
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
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
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [1., 0.]
    ])
    m = ComplexCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    # tf_metric = CategoricalAccuracy()
    # tf_metric.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](.9)
    # assert m.result().numpy() == tf_metric.result().numpy()
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](0.5)
    y_true = np.array([
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [0., 0., 1., 0.],   # 3
        [0., 0., 1., 0.],   # 3
        [0., 0., 1., 0.],   # 3
        [0., 0., 0., 1.],   # 4
        [0., 1., 0., 0.]    # 2
    ])
    y_pred = np.array([
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [1., 0., 0., 0.],   # 1
        [0., 0., 0., 1.],   # 4
        [0., 0., 0., 1.],   # 4
        [0., 0., 0., 1.],   # 4
        [0., 0., 0., 1.],   # 4
        [0., 1., 0., 0.]    # 2
    ])
    m = ComplexAverageAccuracy()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == np.cast['float32'](0.75)


if __name__ == "__main__":
    test_null_label()
    test_with_tf()
    test_metric()
