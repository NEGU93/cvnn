import numpy as np
from cvnn.layers import ComplexDense, ComplexFlatten, ComplexInput, ComplexConv2D, ComplexMaxPooling2D, ComplexAvgPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf

from pdb import set_trace

"""
This module tests:
    Correct result of Complex AVG and MAX pooling layers.
    Init ComplexConv2D layer and verifies output dtype and shape.
    Trains using:
        ComplexDense
        ComplexFlatten
        ComplexInput 
"""


def small_example():
    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i
    c_flat = ComplexFlatten()
    c_dense = ComplexDense(units=10)
    res = c_dense(c_flat(img.astype(np.complex64)))


def serial_layers():
    model = Sequential()
    model.add(ComplexDense(32, activation='relu', input_shape=(32, 32, 3)))
    model.add(ComplexDense(32))
    print(model.output_shape)

    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i

    model = Sequential()
    # model.add(ComplexInput(img.shape[1:]))
    model.add(ComplexFlatten(input_shape=img.shape[1:]))
    model.add(ComplexDense(units=10))

    res = model(img)


def shape_ad_dtype_of_conv2d():
    input_shape = (4, 28, 28, 3)
    x = tf.cast(tf.random.normal(input_shape), tf.complex64)
    y = ComplexConv2D(2, 3, activation='cart_relu', padding="same", input_shape=input_shape[1:], dtype=x.dtype)(x)
    assert y.shape == (4, 28, 28, 2)
    assert y.dtype == tf.complex64


def get_img():
    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ], [
        [0, 4, 5],
        [3, 2, 2],
        [4, 8, 9]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i
    img = np.reshape(img, (2, 3, 3, 1))
    return img


def complex_max_pool_2d():
    img = get_img()
    max_pool = ComplexMaxPooling2D(strides=1)
    res = max_pool(img.astype(np.complex64))
    expected_res = np.array([
        [[
            [2.+7.j],
            [2.+9.j]],
            [[2.+7.j],
            [2.+9.j]]],
       [[
           [7.+2.j],
           [9.+2.j]],
        [
            [5.+8.j],
            [3.+9.j]]
        ]])
    assert (res == expected_res.astype(np.complex64)).numpy().all()
    x = tf.constant([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]])
    x = tf.reshape(x, [1, 3, 3, 1])
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
    complex_max_pool_2d = ComplexMaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
    assert np.all(max_pool_2d(x) == complex_max_pool_2d(x))


def complex_avg_pool():
    img = get_img()
    avg_pool = ComplexAvgPooling2D(strides=1)
    res = avg_pool(img.astype(np.complex64))
    expected_res = np.array([[[[0.75+3.5j], [1.75+6.25j]], [[1.75+4.75j], [4. + 6.j]]],
                             [[[3.5 + 2.25j], [6.25+3.25j]], [[4.75 + 4.25j], [6. + 5.25j]]]])
    assert (res == expected_res.astype(np.complex64)).numpy().all()


def test_layers():
    complex_avg_pool()
    shape_ad_dtype_of_conv2d()
    complex_max_pool_2d()


if __name__ == "__main__":
    test_layers()
