import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.datasets import fashion_mnist
from cvnn.layers import FFT2DTransform, FrequencyConvolutional2D, Flatten, Dense
from cvnn.cvnn_model import CvnnModel
from cvnn.dataset import Dataset
from cvnn.utils import normalize
from pdb import set_trace


"""
img1 = np.array([
    [
        [10, 10, 10, 0, 0, 0],      # R
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0]
    ],
    [
        [20, 20, 20, 0, 0, 0],      # G
        [20, 20, 20, 0, 0, 0],
        [20, 20, 20, 0, 0, 0],
        [20, 20, 20, 0, 0, 0],
        [20, 20, 20, 0, 0, 0],
        [20, 20, 20, 0, 0, 0]
    ],
    [
        [30, 30, 30, 0, 0, 0],      # B
        [30, 30, 30, 0, 0, 0],
        [30, 30, 30, 0, 0, 0],
        [30, 30, 30, 0, 0, 0],
        [30, 30, 30, 0, 0, 0],
        [30, 30, 30, 0, 0, 0]
    ]
]).astype(np.float32)
img2 = img1
img = np.array([img1, img2]).transpose((0, 2, 3, 1))
"""

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels = Dataset.sparse_into_categorical(train_labels, num_classes=10)
test_labels = Dataset.sparse_into_categorical(test_labels, num_classes=10)
train_images = normalize(train_images).astype(np.float32)
test_images = normalize(test_images).astype(np.float32)


shape = [
    FFT2DTransform(input_size=train_images.shape[1:], input_dtype=train_images.dtype, padding=2),
    FrequencyConvolutional2D(filters=1, kernel_shape=(3, 3), activation="cart_relu"),
    Flatten(),
    Dense(output_size=10, activation="softmax_real")
]
model = CvnnModel("Testing_fft", shape, categorical_crossentropy, tensorboard=True, verbose=False)
out = model.call(train_images[:1])
# set_trace()
# model.fit(train_images, train_labels, validation_data=(test_images, test_labels))