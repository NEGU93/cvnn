import tensorflow as tf
import numpy as np
# from cvnn.layers import Convolutional
from pdb import set_trace
import sys
from scipy import signal
from scipy import linalg


COMPARE_TF_AND_NP = False
TWO_DIM_TEST = True
ONE_DIM_TEST = False
STACKOVERFLOW_EXAMPLE = False

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

    # conv = Convolutional(1, (3,), (10, 1), padding=2, input_dtype=np.float32)
    # conv.kernels = []
    # conv.kernels.append(tf.reshape(tf.cast(tf.Variable(c, name="kernel" + str(0) + "_f" + str(0)),
                                           # dtype=np.float32), (3, 1)))
    # std_out = conv([b])[..., 0]

    b_pad = tf.cast(tf.pad(b, tf.constant([[0, 2]])), tf.complex64)
    I = tf.signal.fft(tf.cast(b_pad, tf.complex64))
    paddings = tf.constant([[0, 9]])
    c_pad = tf.cast(tf.pad(c, paddings), tf.complex64)
    C = tf.signal.fft(c_pad)
    F = tf.math.multiply(I, C)
    f = tf.signal.ifft(F)
    f_real = tf.cast(f, tf.int32)

    # print("std_out: " + str(std_out))
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
    mode = 'full'
    # conv = Convolutional(1, (3, 3), (6, 6, 1), padding=2, input_dtype=np.float32)
    # conv.kernels = []
    # conv.kernels.append(tf.reshape(tf.cast(tf.Variable(k, name="kernel" + str(0) + "_f" + str(0)), dtype=np.float32),
    #                                (3, 3, 1)))
    # std_out = conv([img2])[..., 0]
    img2_pad = tf.pad(img2, tf.constant([[0, 2], [0, 2]]))
    k_pad = tf.cast(tf.pad(k, tf.constant([[0, 5], [0, 5]])), tf.complex64)
    I = tf.signal.fft2d(tf.cast(img2_pad, tf.complex64))
    K = tf.signal.fft2d(k_pad)
    F = tf.math.multiply(I, K)
    f = tf.signal.ifft2d(F)
    f_real = tf.cast(f, tf.int32)
    # print("std_out: " + str(std_out))
    print("f_real: " + str(f_real))
    
    
    
    np_fft_conv = np.array(signal.fftconvolve(img2, k, mode=mode) , np.int32)
    print("sp_fft_conv_" + mode + ":\n" + str(np_fft_conv))
    

    np_conv = np.array(signal.convolve2d(img2 , k, mode), np.int32)
    print("sp_fft_conv_" + mode + ":\n" + str(np_conv))
    # set_trace()


    """
    # Check numpy implementation
    I = np.fft.fft2(img2)
    K = np.fft.fft2(tf.pad(k, tf.constant([[0, 5], [0, 5]])))
    F = np.multiply(I, K)
    f = np.fft.ifft2(F)
    print("f_np_real: " + str(np.round(f.astype(np.float32))))
    """
if STACKOVERFLOW_EXAMPLE:
    # https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution

    x = [[1 , 0 , 0 , 0] , [0 , -1 , 0 , 0] , [0 , 0 , 3 , 0] , [0 , 0 , 0 , 1]]
    x = np.array(x)
    y = [[4 , 5] , [3 , 4]]
    y = np.array(y)
    
    print("conv:\n" ,  signal.convolve2d(x , y , 'full'))
    
    s1 = np.array(x.shape)
    s2 = np.array(y.shape)
    
    size = s1 + s2 - 1
    
    
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    
    
    new_x = np.fft.fft2(x , fsize)
    
    
    new_y = np.fft.fft2(y , fsize)
    result = np.fft.ifft2(new_x*new_y)[fslice].copy()
    
    print("fft for my method:\n" , np.array(result.real , np.int32))
    
    print("fft:\n" , np.array(signal.fftconvolve(x ,y) , np.int32))

    
