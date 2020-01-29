import numpy as np
"""-----------
    # Initializers
    # https://keras.io/initializers/
    -----------"""


def glorot_uniform_init(in_neurons, out_neurons):
    return np.random.randn(in_neurons, out_neurons) * np.sqrt(1 / in_neurons)


def rand_init_neg(in_neurons, out_neurons):
    return 2 * np.random.rand(in_neurons, out_neurons) - 1


def rand_init(in_neurons, out_neurons):
    """
    Use this function to make fashion not to predict good
    :param in_neurons:
    :param out_neurons:
    :return:
    """
    return np.random.rand(in_neurons, out_neurons)


__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.1'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
