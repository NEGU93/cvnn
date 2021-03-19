import numpy as np
from cvnn.layers import ComplexDense, ComplexFlatten, ComplexInput, ComplexConv2D, ComplexMaxPooling2D, \
    ComplexAvgPooling2D, ComplexConv2DTranspose, ComplexUnPooling2D
import cvnn.layers as complex_layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow_datasets as tfds

from pdb import set_trace

"""
This module tests:
    Correct result of Complex AVG and MAX pooling layers.
    Init ComplexConv2D layer and verifies output dtype and shape.
    Trains using:
        ComplexDense
        ComplexFlatten
        ComplexInput 
        ComplexDropout
"""


def dense_example():
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
    assert res.shape == [2, 10]
    assert res.dtype == tf.complex64
    model = tf.keras.models.Sequential()
    model.add(ComplexInput(input_shape=(3, 3)))
    model.add(ComplexFlatten())
    model.add(ComplexDense(32, activation='cart_relu'))
    model.add(ComplexDense(32))
    assert model.output_shape == (None, 32)
    res = model(img.astype(np.complex64))


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


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


@tf.autograph.experimental.do_not_convert
def dropout():
    tf.random.set_seed(0)
    layer = complex_layers.ComplexDropout(.2, input_shape=(2,))
    data = np.arange(10).reshape(5, 2).astype(np.float32)
    data = tf.complex(data, data)
    outputs = layer(data, training=True)
    expected_out = np.array([[0. + 0.j, 1.25 + 1.25j],
                             [2.5 + 2.5j, 3.75 + 3.75j],
                             [5. + 5.j, 6.25 + 6.25j],
                             [7.5 + 7.5j, 8.75 + 8.75j],
                             [10. + 10.j, 0. + 0.j]]
                            )
    assert np.all(data == layer(data, training=False))
    assert np.all(outputs == expected_out)
    ds_train, ds_test = get_dataset()
    model = tf.keras.models.Sequential([
        complex_layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32),
        complex_layers.ComplexDense(128, activation='cart_relu', dtype=np.float32),
        complex_layers.ComplexDropout(rate=0.5),
        complex_layers.ComplexDense(10, activation='softmax_real', dtype=np.float32)
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    model.fit(ds_train, epochs=1, validation_data=ds_test, verbose=False, shuffle=False)
    model.evaluate(ds_test, verbose=False)


def get_img():
    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 7, 5],
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


def complex_max_pool_2d(test_unpool=True):
    img = get_img()
    max_pool = ComplexMaxPooling2D(strides=1, data_format="channels_last")
    res = max_pool(img.astype(np.complex64))
    expected_res = np.array([
        [[
            [2. + 7.j],
            [2. + 9.j]],
            [[2. + 7.j],
             [2. + 9.j]]],
        [[
            [7. + 4.j],
            [9. + 2.j]],
            [
                [5. + 8.j],
                [3. + 9.j]]]
    ])
    assert (res.numpy() == expected_res.astype(np.complex64)).all()
    if test_unpool:
        max_unpooling = ComplexUnPooling2D(img.shape[1:])
        unpooled = max_unpooling(res, unpool_mat=max_pool.get_max_index())
        expected_unpooled = np.array([[[0. + 0.j, 0. + 0.j, 0. + 0.j],
                                       [0. + 0.j, 4. + 14.j, 4. + 18.j],
                                       [0. + 0.j, 0. + 0.j, 0. + 0.j]],
                                      [[0. + 0.j, 7. + 4.j, 0. + 0.j],
                                       [0. + 0.j, 0. + 0.j, 9. + 2.j],
                                       [0. + 0.j, 5. + 8.j, 3. + 9.j]]]).reshape(2, 3, 3, 1)
        assert np.all(unpooled.numpy() == expected_unpooled)

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
    expected_res = np.array([[[[0.75 + 3.5j], [1.75 + 6.25j]], [[1.75 + 4.75j], [4. + 6.j]]],
                             [[[4.25 + 2.25j], [7 + 3.25j]], [[4.75 + 4.25j], [6. + 5.25j]]]])
    assert (res.numpy() == expected_res.astype(np.complex64)).all()


def complex_conv_2d_transpose():
    value = [[1, 2, 1], [2, 1, 2], [1, 1, 2]]
    init = tf.constant_initializer(value)
    transpose_2 = ComplexConv2DTranspose(1, kernel_size=3, kernel_initializer=init, dtype=np.float32)
    input = np.array([[55, 52], [57, 50]]).astype(np.float32).reshape((1, 2, 2, 1))
    expected = np.array([
        [55., 162., 159., 52.],
        [167., 323., 319., 154.],
        [169., 264., 326., 204.],
        [57., 107., 164., 100.]
    ], dtype=np.float32)
    assert np.allclose(transpose_2(input).numpy().reshape((4, 4)), expected)  # TODO: Check why the difference
    value = [[1, 2], [2, 1]]
    init = tf.constant_initializer(value)
    transpose_3 = ComplexConv2DTranspose(1, kernel_size=2, kernel_initializer=init, dtype=np.float32)
    expected = np.array([
        [55., 162., 104],
        [167., 323., 152],
        [114., 157, 50]
    ], dtype=np.float32)
    assert np.allclose(transpose_3(input).numpy().reshape((3, 3)), expected)
    complex_transpose = ComplexConv2DTranspose(1, kernel_size=2, dtype=np.complex64)
    complex_input = (input + 1j * np.zeros(input.shape)).astype(np.complex64)
    assert complex_transpose(complex_input).dtype == tf.complex64


@tf.autograph.experimental.do_not_convert
def test_layers():
    complex_conv_2d_transpose()
    complex_max_pool_2d()
    dropout()
    complex_avg_pool()
    shape_ad_dtype_of_conv2d()
    dense_example()


if __name__ == "__main__":
    test_layers()
