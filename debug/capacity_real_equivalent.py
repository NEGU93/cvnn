import numpy as np
import cvnn.layers as layers
from time import sleep
from cvnn.layers import Dense
from cvnn.cvnn_model import CvnnModel
from tensorflow.keras.losses import categorical_crossentropy


def test_shape(input_size, output_size, shape_raw, classifier=True, capacity_equivalent=True):
    layers.ComplexLayer.last_layer_output_dtype = None
    layers.ComplexLayer.last_layer_output_size = None
    shape = [Dense(input_size=input_size, output_size=shape_raw[0] if len(shape_raw) != 0 else output_size,
                   input_dtype=np.complex64)]
    for i in range(1, len(shape_raw)):
        shape.append(Dense(output_size=shape_raw[i], dropout=None))
    if len(shape_raw) != 0:
        shape.append(Dense(output_size=output_size, activation='softmax_real'))

    complex_network = CvnnModel(name="complex_network", shape=shape, loss_fun=categorical_crossentropy,
                                verbose=False, tensorboard=False)
    result = complex_network._get_real_equivalent_multiplier(classifier=classifier,
                                                             capacity_equivalent=capacity_equivalent,
                                                             equiv_technique='alternate')
    # rvnn = complex_network.get_real_equivalent(classifier, capacity_equivalent)
    # complex_network.training_param_summary()
    # rvnn.training_param_summary()
    print(result)


if __name__ == '__main__':
    test_shape(100, 2, [100, 30, 50, 40, 60, 50, 30], classifier=True)
    sleep(2)
    test_shape(100, 2, [100, 30, 50, 60, 50, 30], classifier=True)
    sleep(2)
    test_shape(100, 2, [100, 30, 50, 60, 50, 30], classifier=False)
    sleep(2)
    test_shape(100, 2, [100, 30, 50, 40, 60, 50, 30], classifier=False)
    sleep(2)
    # test_shape(100, 2, [100, 30, 50, 40, 60, 50, 30], capacity_equivalent=False)
    """
    test_shape(100, 2, [])
    sleep(2)
    test_shape(100, 2, [64])
    sleep(2)
    test_shape(100, 2, [100, 64])
    sleep(2)
    test_shape(100, 2, [100, 30, 64])
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 50])
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 60, 30])
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 60, 50, 30])
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 60, 50, 30, 60])

    # Not capacity equivalent
    sleep(2)
    test_shape(100, 2, [], capacity_equivalent=False)
    sleep(2)
    test_shape(100, 2, [64], capacity_equivalent=False)
    sleep(2)
    test_shape(100, 2, [100, 64], capacity_equivalent=False)
    sleep(2)
    test_shape(100, 2, [100, 30, 64], capacity_equivalent=False)
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 50], capacity_equivalent=False)
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 60, 50, 30], capacity_equivalent=False)
    sleep(2)
    test_shape(100, 2, [100, 30, 40, 60, 50, 30], classifier=False, capacity_equivalent=False)
    # sleep(2)
    # test_shape(100, 2, [100, 30, 40, 60, 50, 30], classifier=False)
    """
    """
    [1]
    [2 1]
    [1 2 1]
    [1 2 2 1]
    [1 2 1 2 1]
    [1 2 2 1 2 1]
    [1 2 1 2 1 2 1]
    [1 2 1 2 2 1 2 1]
    [1]
    [2 1]
    [2 2 1]
    [2 2 2 1]
    [2 2 2 2 1]
    [2 2 2 2 2 2 1]
    [2 2 2 2 2 2 2]
    """
