import tensorflow as tf
import tensorflow_datasets as tfds
from cvnn import layers
import numpy as np
import timeit
try:
    import plotly.graph_objects as go
    import plotly
    PLOTLY = True
except ModuleNotFoundError:
    PLOTLY = False

# tf.enable_v2_behavior()
# tfds.disable_progress_bar()

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'editable': True
}


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


def keras_fit(ds_train, ds_test, verbose=True, init1='glorot_uniform', init2='glorot_uniform'):
    tf.random.set_seed(1)
    # https://www.tensorflow.org/datasets/keras_example
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init1),
      tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=init2)
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    start = timeit.default_timer()
    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    stop = timeit.default_timer()
    return history, stop - start


def own_fit(ds_train, ds_test, verbose=True, init1='glorot_uniform', init2='glorot_uniform'):
    tf.random.set_seed(1)
    model = tf.keras.models.Sequential([
        layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32),
        layers.ComplexDense(128, activation='cart_relu', dtype=np.float32, kernel_initializer=init1),
        layers.ComplexDense(10, activation='softmax_real', dtype=np.float32, kernel_initializer=init2)
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    start = timeit.default_timer()
    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    stop = timeit.default_timer()
    return history, stop - start


def test_mnist():
    ds_train, ds_test = get_dataset()
    keras_hist, keras_time = keras_fit(ds_train, ds_test)
    # keras2_hist, keras2_time = keras_fit(ds_train, ds_test)
    own_hist, own_time = own_fit(ds_train, ds_test)
    assert keras_hist.history == own_hist.history, f"{keras_hist.history}\n !=\n {own_hist.history}"
    

if __name__ == "__main__":
    test_mnist()
    # test_mnist_montecarlo()
    # ds_train, ds_test = get_dataset()
    # keras_fit(ds_train, ds_test)
    # own_fit(ds_train, ds_test)


