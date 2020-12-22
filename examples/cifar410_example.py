import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cvnn.layers as complex_layers
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype(dtype=np.float32) / 255.0, test_images.astype(dtype=np.float32) / 255.0

def keras_fit(epochs=10):
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer=init))
    model.add(layers.Dense(10, kernel_initializer=init))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return test_loss, test_acc

def own_fit(epochs=10):
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3), dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2), dtype=np.float32))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2), dtype=np.float32))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexDense(10, dtype=np.float32, kernel_initializer=init))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return test_loss, test_acc

def test_fashion_mnist():
    epochs = 1
    keras = keras_fit(epochs=epochs)
    keras1 = keras_fit(epochs=epochs)
    own = own_fit(epochs=epochs)
    assert keras == own, f"{keras} != {own}"
    
if __name__ == "__main__":
    test_fashion_mnist()