from cvnn.montecarlo import MonteCarlo
import tensorflow as tf
import cvnn.layers as layers
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


own_model = tf.keras.Sequential([
    layers.ComplexFlatten(input_shape=(28, 28)),
    layers.ComplexDense(128, activation='cart_relu', dtype=np.float32),
    layers.ComplexDense(10, dtype=np.float32)
])
own_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

keras_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
keras_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

monte_carlo = MonteCarlo()
monte_carlo.add_model(own_model)
monte_carlo.add_model(keras_model)

monte_carlo.run(x=train_images, y=train_labels, validation_data=(test_images, test_labels), iterations=10)