import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cvnn.layers as complex_layers
import numpy as np
from pdb import set_trace

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype(dtype=np.float32) / 255.0, test_images.astype(dtype=np.float32) / 255.0


def keras_fit(epochs=10):
    tf.random.set_seed(1)
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer=init))
    model.add(layers.Dense(10, kernel_initializer=init))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return history


def own_fit(epochs=10):
    tf.random.set_seed(1)
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3), dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), dtype=np.float32))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), dtype=np.float32))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexDense(10, dtype=np.float32, kernel_initializer=init))
    # model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return history


def test_cifar10():
    epochs = 3
    assert not tf.test.gpu_device_name(), "Using GPU not good for debugging"
    keras = keras_fit(epochs=epochs)
    # keras1 = keras_fit(epochs=epochs)
    own = own_fit(epochs=epochs)
    assert keras.history == own.history, f"\n{keras.history}\n !=\n{own.history}"
    # for k, k2, o in zip(keras.history.values(), keras1.history.values(), own.history.values()):
    #     if np.all(np.array(k) == np.array(k2)):
    #         assert np.all(np.array(k) == np.array(o)), f"\n{keras.history}\n !=\n{own.history}"


if __name__ == "__main__":
    test_cifar10()
