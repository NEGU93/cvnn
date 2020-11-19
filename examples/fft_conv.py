import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from cvnn.layers import FFT2DTransform, FrequencyConvolutional2D
from cvnn.cvnn_model import CvnnModel


img2 = np.array([
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0]
]).astype(np.float32)
img2 = np.reshape(img2, (1, 6, 6, 1))

shape = [
    FFT2DTransform(input_size=img2.shape[1:], input_dtype=img2.dtype, padding=2),
    FrequencyConvolutional2D(filter=1, kernel_shape=(3, 3))
]
model = CvnnModel("Testing_fft", shape, categorical_crossentropy, tensorboard=False, verbose=False)
print(model.call(img2)[0,...,0])