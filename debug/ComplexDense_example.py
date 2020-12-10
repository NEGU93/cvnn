import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets
from pdb import set_trace

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = tf.cast(train_images, tf.complex64) / 255.0, tf.cast(test_images, tf.complex64) / 255.0

model = Sequential([
  layers.Flatten(input_shape=(28, 28, 1)),
  layers.Dense(128, activation='relu', input_shape=(28, 28, 1)),
  layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    train_images, train_labels,
    epochs=6,
    validation_data=(test_images, test_labels),
)