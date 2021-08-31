# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
from cvnn import layers

print(tf.__version__)


def get_fashion_mnist_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def keras_fit(train_images, train_labels, test_images,  test_labels, epochs=10):
    tf.random.set_seed(1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(5, kernel_size=3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    weigths = model.get_weights()
    with tf.GradientTape() as tape:
        loss = model.compiled_loss(y_true=tf.convert_to_tensor(test_labels), y_pred=model(test_images))
        gradients = tape.gradient(loss, model.trainable_weights)  # back-propagation
    history = model.fit(train_images, train_labels, epochs=epochs, shuffle=False)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    logs = {
        'weights_at_init': weigths,
        'loss': loss,
        'gradients': gradients,
        'weights_at_end': model.get_weights()
    }
    return history, logs


def own_fit(train_images, train_labels, test_images,  test_labels, epochs=10):
    tf.random.set_seed(1)
    model = tf.keras.Sequential([
        layers.complex_input(shape=(28, 28, 1), dtype=tf.float32),
        layers.ComplexConv2D(5, kernel_size=3, dtype=tf.float32),
        layers.ComplexBatchNormalization(dtype=tf.float32),
        tf.keras.layers.Activation('relu'),
        layers.ComplexFlatten(),
        layers.ComplexDense(10, dtype=np.float32)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    weigths = model.get_weights()
    with tf.GradientTape() as tape:
        loss = model.compiled_loss(y_true=tf.convert_to_tensor(test_labels), y_pred=model(test_images))
        gradients = tape.gradient(loss, model.trainable_weights)  # back-propagation
    history = model.fit(train_images, train_labels, epochs=epochs, shuffle=False)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    logs = {
        'weights_at_init': weigths,
        'loss': loss,
        'gradients': gradients,
        'weights_at_end': model.get_weights()
    }
    return history, logs


def test_batchnorm_fashion_mnist():
    assert not tf.test.gpu_device_name(), "Using GPU not good for debugging"
    epochs = 3
    (train_images, train_labels), (test_images, test_labels) = get_fashion_mnist_dataset()
    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)
    own, own_logs = own_fit(train_images, train_labels, test_images, test_labels, epochs=epochs)
    keras, keras_logs = keras_fit(train_images, train_labels, test_images, test_labels, epochs=epochs)
    # keras1 = keras_fit(train_images, train_labels, test_images, test_labels, init1=init1, init2=init2, epochs=epochs)
    # if keras.history == keras1.history:
    set_trace()
    assert np.all([np.all(k_w == o_w) for k_w, o_w in zip(keras_logs['weights_at_init'], own_logs['weights_at_init'])])
    assert own_logs['loss'] == keras_logs['loss']  # same loss
    assert np.all([np.allclose(k, o) for k, o in zip(keras_logs['gradients'], own_logs['gradients'][::2])])
    assert keras.history == own.history, f"{keras.history} !=\n{own.history}"


if __name__ == "__main__":
    from importlib import reload
    import os
    import tensorflow

    reload(tensorflow)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    test_batchnorm_fashion_mnist()
