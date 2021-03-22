import tensorflow as tf
from pdb import set_trace
import numpy as np


def nearest_neighbor(input, deisred_size):
    # Put channels first, this will do out[i, j] = in[i, j] even if its a matrix? Sound good
    i_output = None
    j_output = None
    for i in range(0, deisred_size[0]):
        for j in range(0, deisred_size[1]):
            i_new = tf.cast(tf.round((input.shape[0]*i)/deisred_size[0]), dtype=tf.int32)
            j_new = tf.cast(tf.round((input.shape[1]*j)/deisred_size[1]), dtype=tf.int32)
            if j_output is not None:
                j_output = tf.stack([j_output, input[i_new, j_new]], axis=0)
            else:
                j_output = input[i_new, j_new]
        if i_output is not None:
            i_output = tf.stack([i_output, j_output], axis=0)
        else:
            i_output = j_output
        j_output = None
    output = tf.transpose(i_output, perm=[2, 0, 1, 3])
    return output


input_shape = (2, 2, 1, 3)
x = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
z = tf.complex(real=x, imag=x)
z = tf.transpose(z, perm=[1, 2, 0, 3])
nearest_neighbor(z, deisred_size=(2, 2))


