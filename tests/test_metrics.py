import numpy as np
from pdb import set_trace
from cvnn.metrics import ComplexAverageAccuracy, ComplexCategoricalAccuracy


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


if __name__ == "__main__":
    test_metric()
