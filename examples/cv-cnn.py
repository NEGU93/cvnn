from cvnn.layers import Convolutional, MaxPooling, Flatten, Dense
from cvnn.cvnn_model import CvnnModel
from tensorflow.keras.losses import categorical_crossentropy
from time import time
import numpy as np
from pdb import set_trace
from tensorflow.keras import datasets
# https://www.tensorflow.org/tutorials/images/cnn

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0    # Normalize pixel values to be between 0 and 1

model_layers = [Convolutional(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3), input_dtype=np.float32),
                MaxPooling((2, 2)),
                Convolutional(64, (3, 3), activation='cart_relu'),
                MaxPooling((2, 2)),
                Convolutional(64, (3, 3), activation='cart_relu'),
                Flatten(), 
                Dense(64, activation='cart_relu'),
                Dense(10)]

model = CvnnModel("CV-CNN Testing", model_layers, categorical_crossentropy, tensorboard=True)
model.training_param_summary()
# model.fit(train_images, train_labels, epochs=10, verbose=True, save_csv_history=True, fast_mode=True)
