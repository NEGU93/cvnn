import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

tfds.disable_progress_bar()


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
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
    # https://www.tensorflow.org/datasets/keras_example
    tf.random.set_seed(24)
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
    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    return history


def test_mnist():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    init = tf.keras.initializers.GlorotUniform()
    init1 = tf.constant_initializer(init((784, 128)).numpy())
    init2 = tf.constant_initializer(init((128, 10)).numpy())
    ds_train, ds_test = get_dataset()
    keras1 = keras_fit(ds_train, ds_test, init1=init1, init2=init2)
    keras2 = keras_fit(ds_train, ds_test, init1=init1, init2=init2)
    assert keras1.history == keras2.history, f"\n{keras1.history}\n!=\n{keras2.history}"


if __name__ == "__main__":
    test_mnist()