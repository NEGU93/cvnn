import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, models
from cvnn.initializers import ComplexGlorotUniform
from cvnn.layers import ComplexDense, ComplexFlatten, ComplexInput
from cvnn import layers
from cvnn.montecarlo import run_gaussian_dataset_montecarlo


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def mnist_example():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        ComplexFlatten(input_shape=(28, 28, 1)),
        ComplexDense(128, activation='relu', dtype=tf.float32),
        ComplexDense(10, dtype=tf.float32)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=2,
        validation_data=ds_test,
    )


def fashion_mnist_example():
    dtype_1 = np.complex64
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype(dtype_1)
    test_images = test_images.astype(dtype_1)
    train_labels = train_labels.astype(dtype_1)
    test_labels = test_labels.astype(dtype_1)

    model = tf.keras.Sequential([
        ComplexInput(input_shape=(28, 28)),
        ComplexFlatten(),
        ComplexDense(128, activation='cart_relu', kernel_initializer=ComplexGlorotUniform(seed=0)),
        ComplexDense(10, activation='convert_to_real_with_abs', kernel_initializer=ComplexGlorotUniform(seed=0))
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )
    model.fit(train_images, train_labels, epochs=2)
    # import pdb; pdb.set_trace()


def cifar10_test():
    dtype_1 = np.complex64
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.astype(dtype_1)
    test_images = test_images.astype(dtype_1)
    train_labels = train_labels.astype(dtype_1)
    test_labels = test_labels.astype(dtype_1)
    model = models.Sequential()
    model.add(layers.ComplexInput(input_shape=(32, 32, 3), dtype=dtype_1))     # Never forget this!!!
    model.add(layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
    model.add(layers.ComplexAvgPooling2D((2, 2)))
    model.add(layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
    model.add(layers.ComplexMaxPooling2D((2, 2)))       # TODO: This is changing the dtype!
    model.add(layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
    model.add(layers.ComplexFlatten())
    model.add(layers.ComplexDense(64, activation='cart_relu'))
    model.add(layers.ComplexDense(10, activation='convert_to_real_with_abs'))
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=2, validation_data=(test_images, test_labels))


def test_datasets():
    cifar10_test()
    fashion_mnist_example()
    mnist_example()
    run_gaussian_dataset_montecarlo(epochs=2, iterations=1)


if __name__ == '__main__':
    test_datasets()