import numpy as np
import cvnn.layers as complex_layers
import tensorflow as tf
from pdb import set_trace


def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype(dtype=np.complex64) / 255.0
    test_images = test_images.astype(dtype=np.complex64) / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def test_cifar():
    (train_images, train_labels), (test_images, test_labels) = get_dataset()

    # Create your model
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(32, 32, 3)))  # Always use ComplexInput at the start
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_relu'))
    model.add(complex_layers.ComplexDense(10, activation='convert_to_real_with_abs'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.summary()
    history = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


def test_regression():
    input_shape = (4, 28, 28, 3)
    x = tf.cast(tf.random.normal(input_shape), tf.complex64)
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=input_shape[1:]))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(units=64, activation='cart_relu'))
    model.add(complex_layers.ComplexDense(units=10, activation='linear'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y = model(x)
    assert y.dtype == np.complex64


if __name__ == '__main__':
    test_regression()
    test_cifar()
