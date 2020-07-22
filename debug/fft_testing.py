import tensorflow as tf
import numpy as np
from cvnn.layers import Convolutional
from scipy import signal
from scipy import linalg
from pdb import set_trace

COMPARE_TF_AND_NP = False
TWO_DIM_TEST = True
ONE_DIM_TEST = False

if COMPARE_TF_AND_NP:
    # Results are not exactly the same (but fair enough)
    aaa = np.linspace(1.0, 10000.0, 10000)
    x = aaa + 1j * aaa
    x_tensor = tf.convert_to_tensor(x)

    tf_fft = tf.signal.fft(x_tensor)
    np_fft = np.fft.fft(x)

    print(tf_fft.dtype)
    print(np.all(tf_fft.numpy() == np_fft))
    set_trace()

if ONE_DIM_TEST:
    b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    c = [1, 0, 1]

    conv = Convolutional(1, (3,), (10, 1), padding=2, input_dtype=np.float32)
    conv.kernels = []
    conv.kernels.append(tf.reshape(tf.cast(tf.Variable(c, name="kernel" + str(0) + "_f" + str(0)),
                                           dtype=np.float32), (3, 1)))
    std_out = conv([b])[..., 0]

    b_pad = tf.cast(tf.pad(b, tf.constant([[0, 2]])), tf.complex64)
    I = tf.signal.fft(tf.cast(b_pad, tf.complex64))
    paddings = tf.constant([[0, 9]])
    c_pad = tf.cast(tf.pad(c, paddings), tf.complex64)
    C = tf.signal.fft(c_pad)
    F = tf.math.multiply(I, C)
    f = tf.signal.ifft(F)
    f_real = tf.cast(f, tf.int32)

    print("std_out: " + str(std_out))
    print("f_real: " + str(f_real))

if TWO_DIM_TEST:
    img2 = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0]
    ])
    k = [
            [1., 0., -1.],
            [1., 0., -1.],
            [1., 0., -1.]
        ]
    conv = Convolutional(1, (3, 3), (6, 6, 1), padding=2, input_dtype=np.float32)
    conv.kernels = []
    conv.kernels.append(tf.reshape(tf.cast(tf.Variable(k, name="kernel" + str(0) + "_f" + str(0)), dtype=np.float32),
                                   (3, 3, 1)))
    std_out = conv([img2])[..., 0]

    img2 = tf.pad(img2, tf.constant([[0, 2], [0, 2]]))
    I = tf.signal.fft2d(tf.cast(img2, tf.complex64))
    k_pad = tf.cast(tf.pad(k, tf.constant([[0, 5], [0, 5]])), tf.complex64)
    K = tf.signal.fft2d(k_pad)
    F = tf.math.multiply(I, K)
    f = tf.signal.ifft2d(F)
    f_real = tf.cast(f, tf.int32)
    print("std_out: " + str(std_out))
    print("f_real: " + str(f_real))

    np_fft_conv_full = np.array(signal.fftconvolve(img2, k, mode='full') , np.int32)
    print("np_fft_conv_full:\n" + str(np_fft_conv_full))
    
    np_conv = np.array(signal.convolve2d(img2 , k, 'full'), np.int32)
    print("np_conv:\n" + str(np_conv))

    """
    # Check numpy implementation
    I = np.fft.fft2(img2)
    K = np.fft.fft2(tf.pad(k, tf.constant([[0, 5], [0, 5]])))
    F = np.multiply(I, K)
    f = np.fft.ifft2(F)
    print("f_np_real: " + str(np.round(f.astype(np.float32))))
    """

    set_trace()
