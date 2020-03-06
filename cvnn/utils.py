import numpy as np
from datetime import datetime
from pathlib import Path
from pdb import set_trace
import sys
import os
from os.path import join
from scipy.io import loadmat


def load_matlab_matrices(fname="data_cnn1dT.mat", path="/media/barrachina/data/gilles_data/"):
    mat_fname = join(path, fname)
    mat = loadmat(mat_fname)
    return mat


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
__version__ = '0.0.12'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
