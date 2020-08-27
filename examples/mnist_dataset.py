import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from cvnn.cvnn_model import CvnnModel
from cvnn import layers
import numpy as np
from keras.datasets import mnist
from pdb import set_trace
from cvnn.dataset import Dataset

KERAS_DEBUG = False
OWN_MODEL = True

tfds.disable_progress_bar()
tf.enable_v2_behavior()


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def normalize(image):
    return (image / 255.).astype(np.float32)


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
if KERAS_DEBUG:
    # https://www.tensorflow.org/datasets/keras_example
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),  # ATTENTION: The only difference with the link.
        metrics=['accuracy'],
    )
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=2
    )

if OWN_MODEL:
    """(X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    y_test = Dataset.sparse_into_categorical(y_test, 10)
    y_train = Dataset.sparse_into_categorical(y_train, 10)
    print(len(y_test))
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=100)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))"""
    shape = [
        layers.Flatten(input_size=(28, 28, 1), input_dtype=np.float32),
        layers.Dense(output_size=128, activation='cart_relu', input_dtype=np.float32, dropout=None),
        layers.Dense(output_size=10, activation='softmax_real')
    ]
    model = CvnnModel("Testing with MNIST", shape, tf.keras.losses.sparse_categorical_crossentropy,
                      tensorboard=False, verbose=False)
    model.fit(x=ds_train, y=None, validation_data=ds_test, batch_size=128, epochs=6,
              verbose=3, save_csv_history=True, fast_mode=False, save_txt_fit_summary=False)

