import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from cvnn.layers import FFT2DTransform, FrequencyConvolutional2D, Flatten
from cvnn.cvnn_model import CvnnModel
from pdb import set_trace


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

shape = [
    FFT2DTransform(input_size=img.shape[1:], input_dtype=img.dtype, padding=2),
    FrequencyConvolutional2D(filters=1, kernel_shape=(3, 3)),
    Flatten
]
model = CvnnModel("Testing_fft", shape, categorical_crossentropy, tensorboard=False, verbose=False)
print(model.call(img))