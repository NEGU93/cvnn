import numpy as np
import cvnn.layers as layers
from time import sleep
from cvnn.layers import ComplexDense
from cvnn.real_equiv_tools import _get_real_equivalent_multiplier
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy


def shape_tst(input_size, output_size, shape_raw, classifier=True, capacity_equivalent=True, equiv_technique='alternate', expected_result=None):
    shape = [
        layers.ComplexInput(input_shape=input_size, dtype=np.complex64)
    ]
    if len(shape_raw) == 0:
        shape.append(
            ComplexDense(units=output_size, activation='softmax_real', dtype=np.complex64)
        )
    else:  # len(shape_raw) > 0:
        for s in shape_raw:
            shape.append(ComplexDense(units=s, activation='cart_relu'))  # Add dropout!
        shape.append(ComplexDense(units=output_size, activation='softmax_real'))

    complex_network = Sequential(shape, name="complex_network")
    complex_network.compile(optimizer='sgd', loss=categorical_crossentropy, metrics=['accuracy'])
    result = _get_real_equivalent_multiplier(complex_network.layers, classifier=classifier,
                                             capacity_equivalent=capacity_equivalent,
                                             equiv_technique=equiv_technique)
    # rvnn = complex_network.get_real_equivalent(classifier, capacity_equivalent)
    # complex_network.training_param_summary()
    # rvnn.training_param_summary()
    if expected_result is not None:
        assert np.all(expected_result == result), f"Expecting result {expected_result} but got {result}."
    else:
        print(result)


def test_shape():
    # Ratio
    # The bigger the middle, it will tend to sqrt(2) = 1.4142135623730951
    shape_tst(4, 2, [1, 30, 500, 400, 60, 50, 3], classifier=True, equiv_technique='ratio')   
    sleep(2)
    shape_tst(4, 2, [64], classifier=False, equiv_technique='ratio', expected_result=[1, 1])   # this is 1 for regression
    sleep(2)
    shape_tst(4, 2, [64], classifier=True, equiv_technique='ratio', expected_result=[1.2, 1])   # this is 2*(in+out)/(2*in+out) = 1.2
    sleep(2)
    shape_tst(100, 2, [100, 30, 50, 40, 60, 50, 30], classifier=True, equiv_technique='ratio')
    sleep(2)
    shape_tst(100, 2, [100, 30, 50, 60, 50, 30], classifier=True, equiv_technique='ratio')
    sleep(2)
    shape_tst(100, 2, [100, 30, 50, 60, 50, 30], classifier=False, equiv_technique='ratio')
    sleep(2)
    shape_tst(100, 2, [100, 30, 50, 40, 60, 50, 30], classifier=False, equiv_technique='ratio')
    sleep(2)
    shape_tst(100, 2, [100, 30, 50, 40, 60, 50, 30], capacity_equivalent=False, equiv_technique='ratio')
    
    # Alternate
    sleep(2)
    shape_tst(100, 2, [], expected_result=[1])
    sleep(2)
    shape_tst(100, 2, [64], expected_result=[204/202, 1])
    sleep(2)
    shape_tst(100, 2, [100, 64], expected_result=[1, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 64], expected_result=[1, 328/228, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 50], expected_result=[1, 2, 1, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 60, 30], expected_result=[1, 2, 180/120, 1, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 60, 50, 30], expected_result=[1, 2, 1, 2, 1, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 60, 50, 30, 60], expected_result=[1, 2, 1, 180/140, 2, 1, 2, 1])

    # Not capacity equivalent
    sleep(2)
    shape_tst(100, 2, [], capacity_equivalent=False, expected_result=[1])
    sleep(2)
    shape_tst(100, 2, [64], capacity_equivalent=False, expected_result=[2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 64], capacity_equivalent=False, expected_result=[2, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 64], capacity_equivalent=False, expected_result=[2, 2, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 50], capacity_equivalent=False, expected_result=[2, 2, 2, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 60, 50, 30], capacity_equivalent=False, expected_result=[2, 2, 2, 2, 2, 2, 1])
    sleep(2)
    shape_tst(100, 2, [100, 30, 40, 60, 50, 30], classifier=False, capacity_equivalent=False,
               expected_result=[2, 2, 2, 2, 2, 2, 2])
    # sleep(2)
    # shape_tst(100, 2, [100, 30, 40, 60, 50, 30], classifier=False)"""
