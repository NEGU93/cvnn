import numpy as np
from datetime import datetime
from pathlib import Path
from pdb import set_trace
import sys
import os


def create_folder(root_path, now=None):
    if now is None:
        now = datetime.today()
    path = Path(__file__).parents[1].absolute() / Path(root_path + now.strftime("%Y/%m%B/%d%A/run-%Hh%Mm%S/"))
    os.makedirs(path, exist_ok=True)        # Do this not to have a problem if I run in parallel
    return path


def get_func_name(fun):
    if callable(fun):
        return fun.__name__
    elif isinstance(fun, str):
        return fun
    else:
        sys.exit("Error::_get_func_name: Function not recognizable")


def transform_to_real(x_complex):
    """
    :param x_complex: Complex-valued matrix of size mxn
    :return: real-valued matrix of size mx(2*n) unwrapping the real and imag part of the complex-valued input matrix
    """
    # import pdb; pdb.set_trace()
    m = np.shape(x_complex)[0]
    n = np.shape(x_complex)[1]
    x_real = np.ones((m, 2*n))
    x_real[:, :n] = np.real(x_complex)
    x_real[:, n:] = np.imag(x_complex)
    dtype = x_real.dtype
    if x_complex.dtype == np.complex64:
        dtype = np.float32
    elif x_complex.dtype == np.complex128:
        dtype = np.float64
    else:
        print("Warning: transform_to_real: data type unknown: " + str(np.dtype(x_complex)))
    return x_real.astype(dtype)


def cart2polar(z):
    return np.abs(z), np.angle(z)


def polar2cart(rho, angle):
    return rho * np.exp(1j*angle)


def randomize(x, y):
    """
    Randomizes the order of data samples and their corresponding labels
    :param x: data
    :param y: data labels
    :return: Tuple of (shuffled_x, shuffled_y) maintaining coherence of elements labels
    """
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    """
    Get next batch from x and y using start and end
    :param x: data
    :param y: data labels
    :param start: starting index of the batch to be returned
    :param end: end index of the batch to be returned (not including)
    :return: tuple (x, y) of the selected batch
    """
    if start < 0:
        sys.exit("Error:get_next_batch(): start parameter cannot be negative")
    if start > end:     # What will happen if not? Should I leave this case anyway and just give a warning?
        sys.exit("Error:get_next_batch(): end should be higher than start")
    # TODO: Check end < len(x)
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def normalize(x):
    return (x-np.amin(x))/np.abs(np.amax(x)-np.amin(x))     # Checked it works for complex values


def tensorflow_argmax_np_equivalent(x, num_classes):
    res = np.zeros((np.argmax(x, 1).shape[0], num_classes))
    indx = 0
    for k in np.argmax(x, 1):
        res[indx, k] = 1
        indx += 1
    return res


def compute_accuracy(x, y):
    return np.average(np.equal(x, y).all(axis=1))


__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.10'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
