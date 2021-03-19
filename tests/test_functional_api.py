from cvnn.layers import ComplexUnPooling2D, complex_input, ComplexMaxPooling2DWithArgmax
import tensorflow as tf
import numpy as np
from pdb import set_trace


def get_img():
    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 7, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ], [
        [0, 4, 5],
        [3, 2, 2],
        [4, 8, 9]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i
    img = np.reshape(img, (2, 3, 3, 1))
    return img


def unpooling_example():
    x = get_img()
    inputs = complex_input(shape=x.shape[1:])
    max_pool_o, max_arg = ComplexMaxPooling2DWithArgmax(strides=1, data_format="channels_last", name="argmax")(inputs)
    # max_pool_o = ComplexMaxPooling2D(strides=1, data_format="channels_last")(inputs)
    max_unpool = ComplexUnPooling2D(x.shape[1:])
    outputs = max_unpool([max_pool_o, max_arg])
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pooling_model")
    model.summary()
    model(x)
    print(model(x)[..., 0])
    # set_trace()
    return model


def test_functional_api():
    model = unpooling_example()


if __name__ == "__main__":
    test_functional_api()