import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import cvnn.optimizers as optimizers
from cvnn.cvnn_model import CvnnModel
from cvnn import layers
import numpy as np
from tqdm import tqdm
from pdb import set_trace
import plotly.graph_objects as go

tfds.disable_progress_bar()
tf.enable_v2_behavior()


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def normalize(image):
    return (image / 255.).astype(np.float32)


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
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def keras_fit(ds_train, ds_test, verbose):
    # https://www.tensorflow.org/datasets/keras_example
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(),  # ATTENTION: The only difference with the link.
        metrics=['accuracy'],
    )
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose
    )
    return model.evaluate(ds_test, verbose=verbose)


def own_fit(ds_train, ds_test, verbose):
    shape = [
        layers.Flatten(input_size=(28, 28, 1), input_dtype=np.float32),
        layers.Dense(output_size=128, activation='cart_relu', input_dtype=np.float32, dropout=None),
        layers.Dense(output_size=10, activation='softmax_real')
    ]
    model = CvnnModel("Testing with MNIST", shape, tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=optimizers.SGD(),
                      tensorboard=False, verbose=False)
    model.fit(x=ds_train, y=None, validation_data=ds_test, batch_size=128, epochs=6,
              verbose=verbose, save_csv_history=True)
    return model.evaluate(ds_test)


if __name__ == "__main__":
    # Parameters
    KERAS_DEBUG = True
    OWN_MODEL = True
    iterations = 10
    verbose = 0

    # Training
    ds_train, ds_test = get_dataset()
    keras_results = []
    own_results = []
    for it in tqdm(range(iterations)):
        if KERAS_DEBUG:
            keras_results.append(keras_fit(ds_train, ds_test, verbose=verbose)[1])
        if OWN_MODEL:
            own_results.append(own_fit(ds_train, ds_test, verbose=verbose)[1])
    fig = go.Figure()
    fig.add_trace(go.Box(y=keras_results, name='Keras'))
    fig.add_trace(go.Box(y=own_results, name='Own'))
    fig.show()

