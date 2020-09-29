import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import cvnn.optimizers as optimizers
from cvnn.cvnn_model import CvnnModel
from cvnn import layers
import numpy as np
from tqdm import tqdm
from pdb import set_trace
import plotly.graph_objects as go
import plotly

tfds.disable_progress_bar()
tf.enable_v2_behavior()

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'editable': True
}


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


def keras_fit(ds_train, ds_test, verbose,
              optimizer_name="Adam"):
    # https://www.tensorflow.org/datasets/keras_example
    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam()
    else:
        optimizer = tf.keras.optimizers.SGD()
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,  # ATTENTION: The only difference with the link.
        metrics=['accuracy'],
    )
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose
    )
    return model.evaluate(ds_test, verbose=verbose)


def own_fit(ds_train, ds_test, verbose,
            optimizer_name="Adam",
            activation_1=tf.keras.activations.relu, activation_2=tf.keras.activations.softmax,
            weight_initializer=tf.keras.initializers.GlorotUniform(), bias_initializer=tf.keras.initializers.Zeros()):
    if optimizer_name == "Adam":
        optimizer = optimizers.Adam()
    else:
        optimizer = optimizers.SGD()
    shape = [
        layers.Flatten(input_size=(28, 28, 1), input_dtype=np.float32),
        layers.Dense(output_size=128,
                     activation=activation_1,
                     input_dtype=np.float32, dropout=None,
                     weight_initializer=weight_initializer,
                     bias_initializer=bias_initializer
                     ),
        layers.Dense(output_size=10,
                     activation=activation_2,
                     weight_initializer=weight_initializer,
                     bias_initializer=bias_initializer
                     )
    ]
    model = CvnnModel("Testing with MNIST", shape, tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=optimizer,
                      tensorboard=False, verbose=False)
    model.fit(x=ds_train, y=None, validation_data=ds_test, epochs=6,
              verbose=verbose, save_csv_history=True)
    return model.evaluate(ds_test)


def test_mnist_montecarlo(optimizer_name='Adam'):
    # Parameters
    KERAS_DEBUG = True
    OWN_MODEL = True
    iterations = 1000
    verbose = 1 if iterations == 1 else 0

    # Training
    ds_train, ds_test = get_dataset()
    keras_results = []
    own_results = []
    activation_1 = tf.keras.activations.relu
    activation_2 = tf.keras.activations.softmax
    weight_initializer = tf.keras.initializers.GlorotUniform()
    bias_initializer = tf.keras.initializers.Zeros()
    fig = go.Figure()

    if KERAS_DEBUG:
        tf.print("Keras simulation")
        for _ in tqdm(range(iterations)):
            keras_results.append(keras_fit(ds_train, ds_test, verbose=verbose, optimizer_name=optimizer_name)[1])
        fig.add_trace(go.Box(y=keras_results, name='Keras'))
    if OWN_MODEL:
        tf.print("Cvnn simulation")
        for _ in tqdm(range(iterations)):
            own_results.append(own_fit(ds_train, ds_test, verbose=verbose, optimizer_name=optimizer_name,
                                       activation_1=activation_1, activation_2=activation_2,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)[1])
        fig.add_trace(go.Box(y=own_results, name='Cvnn'))
    plotly.offline.plot(fig, filename="./results/" + optimizer_name + "_mnist_test.html",
                        config=PLOTLY_CONFIG, auto_open=True)


if __name__ == "__main__":
    test_mnist_montecarlo("SGD")
    test_mnist_montecarlo()


