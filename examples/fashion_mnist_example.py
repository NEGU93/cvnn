# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from cvnn import layers

print(tf.__version__)


def get_fashion_mnist_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def keras_fit(train_images, train_labels, test_images,  test_labels, 
              init1='glorot_uniform', init2='glorot_uniform', epochs=10):
    tf.random.set_seed(1)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init1),
        tf.keras.layers.Dense(10, kernel_initializer=init2)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, shuffle=False)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    return history


def own_fit(train_images, train_labels, test_images,  test_labels, 
            init1='glorot_uniform', init2='glorot_uniform', epochs=10):
    tf.random.set_seed(1)
    model = tf.keras.Sequential([
        layers.ComplexFlatten(input_shape=(28, 28)),
        layers.ComplexDense(128, activation='cart_relu', dtype=np.float32, kernel_initializer=init1),
        layers.ComplexDense(10, dtype=np.float32, kernel_initializer=init2)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, shuffle=False)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    return history


def test_fashion_mnist():
    seed = 117
    epochs = 3
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    init1 = tf.constant_initializer(init((784, 128)).numpy())
    init2 = tf.constant_initializer(init((128, 10)).numpy())
    (train_images, train_labels), (test_images, test_labels) = get_fashion_mnist_dataset()
    keras = keras_fit(train_images, train_labels, test_images, test_labels, init1=init1, init2=init2, epochs=epochs)
    # keras1 = keras_fit(train_images, train_labels, test_images, test_labels, init1=init1, init2=init2, epochs=epochs)
    own = own_fit(train_images, train_labels, test_images, test_labels, init1=init1, init2=init2, epochs=epochs)
    assert keras.history == own.history, f"{keras.history } != {own.history }"


if __name__ == "__main__":
    test_fashion_mnist()
