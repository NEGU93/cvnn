import numpy as np


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def create_data(m, n, mu, sigma):
    """
    Creates a numpy matrix of size mxn with random gaussian distribution of mean mu and variance sigma
    """
    x = mu + 1j*mu + sigma * (np.random.rand(m, n) + 1j * np.random.rand(m, n))
    return x


def create_non_correlated_gaussian_noise(m, n, num_classes=2):
    """

    :param m: Number of examples per class
    :param n: Size of vector
    :param num_classes: Number of different classes to be made
    :return: tuple of a (num_classes*m)xn matrix with data and labels regarding it class.
    """
    x = np.ones((num_classes*m, n)) + 1j*np.ones((num_classes*m, n))
    y = np.ones((num_classes*m, 1))
    for k in range(num_classes):
        mu = int(100*np.random.rand())
        sigma = int(10*np.random.rand())
        x[k*m:(k+1)*m, :] = create_data(m, n, mu, sigma)
        y[k*m:(k+1)*m] = k * y[k*m:(k+1)*m]

    return x, y


def transform_to_real(x_complex):
    """
    :param x_complex: Complex-valued matrix of size mxn
    :return: real-valued matrix of size mx(2*n) unwrapping the real and imag part of the complex-valued input matrix
    """
    m = np.shape(x_complex)[0]
    n = np.shape(x_complex)[1]
    x_real = np.ones((m, 2*n))
    x_real[:, :n] = np.real(x_complex)
    x_real[:, n:] = np.imag(x_complex)
    return x_real


def separate_into_train_and_test(x, y, ratio=0.8):
    m = np.shape(x)[0]
    x_train = x[:int(m*0.8)]
    y_train = y[:int(m*0.8)]
    x_test = x[int(m*0.8):]
    y_test = y[int(m*0.8):]
    return x_train, y_train, x_test, y_test


def get_non_correlated_gaussian_noise(m, n, num_classes=2):
    x, y = create_non_correlated_gaussian_noise(m, n, num_classes)
    x, y = randomize(x, y)
    return separate_into_train_and_test(x, y)
