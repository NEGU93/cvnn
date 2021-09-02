from cvnn.utils import transform_to_real_map_function
import numpy as np
from pdb import set_trace
import tensorflow as tf


def test_image_real_conversion():
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
    label = img

    dataset = tf.data.Dataset.from_tensor_slices((img, label))
    real_dataset = dataset.map(transform_to_real_map_function)
    c_elem, c_label = next(iter(dataset))
    r_elem, r_label = next(iter(real_dataset))

    assert np.all(c_elem.shape[:-1] == r_elem.shape[:-1])
    assert 2 * c_elem.shape[-1] == r_elem.shape[-1]
    assert np.all(tf.math.real(c_elem)[:, :, 0] == r_elem[:, :, 0])
    assert np.all(tf.math.imag(c_elem)[:, :, 0] == r_elem[:, :, 1])


if __name__ == '__main__':
    test_image_real_conversion()

