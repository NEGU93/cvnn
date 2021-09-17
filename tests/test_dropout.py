import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pdb import set_trace
import cvnn.layers as complex_layers
from cvnn.montecarlo import run_montecarlo


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
def simple_random_example():
    tf.random.set_seed(0)
    layer = complex_layers.ComplexDropout(.2, input_shape=(2,), seed=0)
    data = np.arange(10).reshape(5, 2).astype(np.float32)
    data = tf.complex(data, data)
    outputs = layer(data, training=True)
    expected_out = np.array([[0. + 0.j, 0. + 0.j],
                             [0. + 0.j, 3.75 + 3.75j],
                             [5. + 5.j, 6.25 + 6.25j],
                             [7.5 + 7.5j, 8.75 + 8.75j],
                             [10. + 10.j, 11.25 + 11.25j]])
    assert np.all(data == layer(data, training=False))
    assert np.all(outputs == expected_out)
    tf.random.set_seed(0)
    layer = tf.keras.layers.Dropout(.2, input_shape=(2,), seed=0)
    real_outputs = layer(tf.math.real(data), training=True)
    assert np.all(real_outputs == tf.math.real(outputs))


def get_real_mnist_model():
    in1 = tf.keras.layers.Input(shape=(28, 28, 1))
    flat = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(in1)
    dense = tf.keras.layers.Dense(128, activation='cart_relu')(flat)
    # drop = complex_layers.ComplexDropout(rate=0.5)(dense)
    drop = tf.keras.layers.Dropout(0.5)(dense)
    out = tf.keras.layers.Dense(10, activation='softmax_real_with_abs', kernel_initializer="ComplexGlorotUniform")(drop)
    real_model = tf.keras.Model(in1, out, name="tf_rvnn")
    real_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    real_intermediate_model = tf.keras.Model(in1, drop)
    return real_model, real_intermediate_model


def get_complex_mnist_model():
    inputs = complex_layers.complex_input(shape=(28, 28, 1), dtype=np.float32)
    flat = complex_layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32)(inputs)
    dense = complex_layers.ComplexDense(128, activation='cart_relu', dtype=np.float32)(flat)
    drop = complex_layers.ComplexDropout(rate=0.5)(dense)
    out = complex_layers.ComplexDense(10, activation='softmax_real_with_abs', dtype=np.float32)(drop)
    complex_model = tf.keras.Model(inputs, out, name="rvnn")
    complex_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )

    complex_intermediate_model = tf.keras.Model(inputs, drop)
    return complex_model, complex_intermediate_model


def dropout():
    ds_train, ds_test = get_dataset()
    train_images, train_labels = convert_to_numpy(ds_train)
    test_images, test_labels = convert_to_numpy(ds_test)
    img, label = next(iter(ds_test))

    tf.random.set_seed(0)
    complex_model, complex_intermediate_model = get_complex_mnist_model()
    tf.random.set_seed(0)
    real_model, real_intermediate_model = get_real_mnist_model()

    c_before_train_eval = complex_intermediate_model(img, training=False)
    r_before_train_eval = real_intermediate_model(img, training=False)
    assert np.all(r_before_train_eval == c_before_train_eval), f"Results are not equal after drop with training=False"
    assert np.all(real_model.layers[2].get_weights()[0] == complex_model.layers[2].get_weights()[
        0]), f"Output layer weights are not equal before any call"
    assert np.all(real_model.layers[-1].get_weights()[0] == complex_model.layers[-1].get_weights()[
        0]), f"Output layer weights are not equal before any call"
    c_before_train_eval = complex_model(img, training=False)
    r_before_train_eval = real_model(img, training=False)
    assert np.all(r_before_train_eval == c_before_train_eval), f"Results are not equal with training=False"

    tf.random.set_seed(0)
    c_before_train_eval = complex_intermediate_model(img, training=True)
    tf.random.set_seed(0)
    r_before_train_eval = real_intermediate_model(img, training=True)
    assert np.all(r_before_train_eval == c_before_train_eval), f"Results are not equal after drop with training=True"

    tf.random.set_seed(0)
    c_before_train_eval = complex_model(img, training=True)
    tf.random.set_seed(0)
    r_before_train_eval = real_model(img, training=True)
    assert np.all(r_before_train_eval == c_before_train_eval), f"Results are not equal with training=True"

    tf.random.set_seed(0)
    complex_eval = complex_model.evaluate(ds_test, verbose=False)
    tf.random.set_seed(0)
    real_eval = real_model.evaluate(ds_test, verbose=False)
    assert np.all(real_eval == complex_eval), f"\n{real_eval}\n !=\n{complex_eval}"
    elem, label = convert_to_numpy(ds_test)
    label = tf.convert_to_tensor(label)
    # elem, label = next(iter(ds_test))
    # set_trace()
    tf.random.set_seed(0)
    with tf.GradientTape() as tape:
        r_loss = real_model.compiled_loss(y_true=label, y_pred=real_model(elem, training=True))  # calculate loss
        real_gradients = tape.gradient(r_loss, real_model.trainable_weights)  # back-propagation
    tf.random.set_seed(0)
    with tf.GradientTape() as tape:
        c_loss = complex_model.compiled_loss(y_true=label, y_pred=complex_model(elem, training=True))  # calculate loss
        complex_gradients = tape.gradient(c_loss, complex_model.trainable_weights)  # back-propagation
    assert r_loss == c_loss, f"\nReal loss:\t\t {r_loss};\nComplex loss:\t {c_loss}"
    # Next assertions showed a rounding error with my library.
    assert np.all([np.allclose(c_g, r_g) for c_g, r_g in zip(complex_gradients, real_gradients)])


def convert_to_numpy(ds):
    ds_numpy = tfds.as_numpy(ds)
    train_images = None
    train_labels = None
    for ex in ds_numpy:
        if train_images is None:
            train_images = ex[0]
            train_labels = ex[1]
        else:
            train_images = np.concatenate((train_images, ex[0]), axis=0)
            train_labels = np.concatenate((train_labels, ex[1]), axis=0)
    return train_images, train_labels


def mnist(tf_data: bool = True):
    ds_train, ds_test = get_dataset()
    train_images, train_labels = convert_to_numpy(ds_train)
    test_images, test_labels = convert_to_numpy(ds_test)
    tf.random.set_seed(0)
    complex_model, _ = get_complex_mnist_model()
    tf.random.set_seed(0)
    real_model, _ = get_real_mnist_model()
    if tf_data:
        r_history = real_model.fit(ds_train, epochs=6, validation_data=ds_test,
                                   verbose=False, shuffle=False)
        c_history = complex_model.fit(ds_train, epochs=6, validation_data=ds_test,
                                      verbose=False, shuffle=False)
    else:
        r_history = real_model.fit(train_images, train_labels, epochs=6, validation_data=(test_images, test_labels),
                                   verbose=False, shuffle=False)
        c_history = complex_model.fit(train_images, train_labels, epochs=6, validation_data=(test_images, test_labels),
                                      verbose=False, shuffle=False)
    assert r_history.history == c_history.history, f"{r_history.history} != {c_history.history}"


def get_fashion_mnist_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = get_fashion_mnist_dataset()
    tf.random.set_seed(0)
    complex_model, _ = get_complex_mnist_model()
    tf.random.set_seed(0)
    real_model, _ = get_real_mnist_model()

    c_history = complex_model.fit(train_images, train_labels, epochs=10, shuffle=True, verbose=False,
                                  validation_data=(test_images, test_labels))
    r_history = real_model.fit(train_images, train_labels, epochs=10, shuffle=True, verbose=False,
                               validation_data=(test_images, test_labels))
    assert r_history.history == c_history.history, f"{r_history.history} != {c_history.history}"


def montecarlo():
    ds_train, ds_test = get_dataset()
    complex_model, _ = get_complex_mnist_model()
    real_model, _ = get_real_mnist_model()
    run_montecarlo(models=[complex_model, real_model], dataset=ds_train, iterations=30,
                   epochs=20, validation_data=ds_test, do_all=True, validation_split=0.0, preprocess_data=False)


def test_dropout():
    from importlib import reload
    import os
    import tensorflow
    reload(tensorflow)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    dropout()
    mnist(True)
    mnist(False)
    fashion_mnist()
    simple_random_example()


if __name__ == "__main__":
    test_dropout()
