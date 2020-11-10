from cvnn.layers import Convolutional, MaxPooling, Flatten, Dense, AvgPooling
from cvnn.cvnn_model import CvnnModel
from cvnn.dataset import Dataset
from tensorflow.keras.losses import categorical_crossentropy
from time import time
import numpy as np
from pdb import set_trace
from tensorflow.keras import datasets
# https://www.tensorflow.org/tutorials/images/cnn

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0    # Normalize pixel values to be between 0 and 1
train_labels = Dataset.sparse_into_categorical(train_labels)
test_labels = Dataset.sparse_into_categorical(test_labels)

"""
    Convolutional(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3), input_dtype=np.float32),
    AvgPooling((2, 2)),
    Convolutional(64, (3, 3), activation='cart_relu'),
    AvgPooling((2, 2)),
    Convolutional(64, (3, 3), activation='cart_relu'),
    Flatten(),
    """
# train_images = np.reshape(train_images, (train_images.shape[0], np.prod(train_images.shape[1:])))
# train_labels = np.reshape(train_labels, (train_labels.shape[0], np.prod(train_labels.shape[1:])))
model_layers = [
    Convolutional(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3), input_dtype=np.float32),
    Flatten(),
    # Flatten(input_dtype=np.float32, input_size=train_images.shape[1:]),
    # Dense(64, activation='cart_relu', input_dtype=np.float32, input_size=train_images.shape[1]),
    Dense(64, activation='cart_relu'),
    Dense(10, activation='softmax_real')
]

model = CvnnModel("CV-CNN Testing", model_layers, categorical_crossentropy, tensorboard=False, verbose=False)
model.training_param_summary()
model.fit(train_images[:10].astype(np.float32), train_labels[:10].astype(np.float32), validation_split=0.2,
          epochs=2, batch_size=2,
          verbose=True, save_csv_history=False)
