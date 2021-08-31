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


def test_functional_api():
    inputs = complex_layers.complex_input(shape=(128, 128, 3))
    c0 = complex_layers.ComplexConv2D(32, activation='cart_relu', kernel_size=3)(inputs)
    c1 = complex_layers.ComplexConv2D(32, activation='cart_relu', kernel_size=3)(c0)
    c2 = complex_layers.ComplexMaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)
    t01 = complex_layers.ComplexConv2DTranspose(5, kernel_size=2, strides=(2, 2), activation='cart_relu')(c2)
    concat01 = tf.keras.layers.concatenate([t01, c1], axis=-1)

    c3 = complex_layers.ComplexConv2D(4, activation='cart_relu', kernel_size=3)(concat01)
    out = complex_layers.ComplexConv2D(4, activation='cart_relu', kernel_size=3)(c3)
    model = tf.keras.Model(inputs, out)


if __name__ == '__main__':
    test_functional_api()
    test_regression()
    test_cifar()
