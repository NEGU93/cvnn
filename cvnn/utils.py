import numpy as np
from datetime import datetime
from pathlib import Path
from pdb import set_trace
import sys
from tensorflow.python.keras import Model
import tensorflow as tf     # TODO: Imported only for dtype
import os
from os.path import join
from scipy.io import loadmat
# To test logger:
import cvnn
import logging
from typing import Type

logger = logging.getLogger(cvnn.__name__)

REAL_CAST_MODES = {
    'real_imag': 2,
    'amplitude_phase': 2,
    'amplitude_only': 1,
    'real_only': 1
}


def reset_weights(model: Type[Model]):
    # https://github.com/keras-team/keras/issues/341#issuecomment-539198392
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))


def load_matlab_matrices(fname="data_cnn1dT.mat", path="/media/barrachina/data/gilles_data/"):
    """
    Opens Matlab matrix (.mat) as numpy array.
    :param fname: file name to be opened
    :param path: path to file
    :return: numpy array with the Matlab matrix information
    """
    mat_fname = join(path, fname)
    mat = loadmat(mat_fname)
    return mat


def create_folder(root_path, now=None):
    """
    Creates folders within root_path using a date format.
    :param root_path: root path where to create the folder chain
    :param now: date to be used. If None then it will use current time
    :return: the created path in pathlib format (compatible across different OS)
    """
    if now is None:
        now = datetime.today()
    # path = Path(__file__).parents[1].absolute() / Path(root_path + now.strftime("%Y/%m%B/%d%A/run-%Hh%Mm%S/"))
    # Last line was to create inside cvnn. I prefer now to save stuff on each project folder and not on libraries folder
    path = Path(root_path + now.strftime("%Y/%m%B/%d%A/run-%Hh%Mm%S/"))
    os.makedirs(path, exist_ok=True)        # Do this not to have a problem if I run in parallel
    return path


def cast_to_path(path):
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        logger.error("Path datatype not recognized")
        sys.exit(-1)
    return path


def get_func_name(fun):
    """
    Returns the name of a function passed as parameter being either a function itself or a string with the function name
    :param fun: function or function name
    :return: function name
    """
    if callable(fun):
        return fun.__name__
    elif isinstance(fun, str):
        return fun
    else:
        logger.error("Function not recognizable", stack_info=True)
        sys.exit(-1)


def transform_to_real_map_function(image, label, mode: str = "real_imag"):
    if mode not in REAL_CAST_MODES:
        raise KeyError(f"Unknown real cast mode {mode}")
    if mode == 'real_imag':
        ret_value = tf.concat([tf.math.real(image), tf.math.imag(image)], axis=-1)
    elif mode == 'amplitude_phase':
        ret_value = tf.concat([tf.math.abs(image), tf.math.angle(image)], axis=-1)
    elif mode == 'amplitude_only':
        ret_value = tf.math.abs(image)
    elif mode == 'real_only':
        ret_value = tf.math.real(image)
    else:
        raise KeyError(f"Real cast mode {mode} not implemented")
    return ret_value, label


def transform_to_real(x_complex, mode: str = "real_imag"):
    """
    Transforms a complex input matrix into a real value matrix (double size)
    :param x_complex: Complex-valued matrix of size mxn
    :param mode: Mode on how to transform to real. One of the following:
        - real_imag: Separate x_complex into real and imaginary making the size of the return double x_complex
        - amplitude_phase: Separate x_complex into amplitude and phase making the size of the return double x_complex
        - amplitude_only: Apply the absolute value to x_complex. Shape remains the same.
    :return: real-valued matrix of real valued cast of x_complex
    """
    # import pdb; pdb.set_trace()
    if not tf.dtypes.as_dtype(x_complex.dtype).is_complex:
        # Intput was not complex, nothing to do
        return x_complex
    if mode not in REAL_CAST_MODES:
        raise KeyError(f"Unknown real cast mode {mode}")
    m = np.shape(x_complex)[0]
    n = np.prod(np.shape(x_complex)[1:])
    flat_x_complex = np.reshape(x_complex, (m, n))
    multiplier = REAL_CAST_MODES[mode]
    x_real = np.ones((m, multiplier * n))
    if mode == 'real_imag':
        x_real[:, :n] = np.real(flat_x_complex)
        x_real[:, n:] = np.imag(flat_x_complex)
    elif mode == 'amplitude_phase':
        x_real[:, :n] = np.abs(flat_x_complex)
        x_real[:, n:] = np.angle(flat_x_complex)
    elif mode == 'amplitude_only':
        x_real = np.abs(flat_x_complex)
    else:
        raise KeyError(f"Real cast mode {mode} not implemented")
    return np.reshape(x_real, np.shape(x_complex)[:-1] + (np.shape(x_complex)[-1]*multiplier,))


def cart2polar(z):
    """
    :param z: complex input
    :return: tuple with the absolute value of the input and the phase
    """
    return np.abs(z), np.angle(z)


def polar2cart(rho, angle):
    """
    :param rho: absolute value
    :param angle: phase
    :return: complex number using phase and angle
    """
    return rho * np.exp(1j*angle)


def randomize(x, y):
    """
    Randomizes the order of data samples and their corresponding labels
    :param x: data
    :param y: data labels
    :return: Tuple of (shuffled_x, shuffled_y) maintaining coherence of elements labels
    """
    if isinstance(x, tf.data.Dataset):
        return x.shuffle(1000), y
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def normalize(x):
    return (x-np.amin(x))/np.abs(np.amax(x)-np.amin(x))     # Checked it works for complex values


def standarize(x):
    return (x - np.mean(x)) / np.std(x)


def tensorflow_argmax_np_equivalent(x, num_classes):
    res = np.zeros((np.argmax(x, 1).shape[0], num_classes))
    indx = 0
    for k in np.argmax(x, 1):
        res[indx, k] = 1
        indx += 1
    return res


def compute_accuracy(x, y):
    return np.average(np.equal(x, y).all(axis=1))


def median_error(q_75: float, q_25: float, n: int) -> int:
    assert q_75 >= q_25 >= 0.0, f"q_75 {q_75} < q_25 {q_25}"
    return 1.57*(q_75-q_25)/np.sqrt(n)


if __name__ == "__main__":
    logger.warning("Testing logger")


__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.26'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
