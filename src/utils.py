import tensorflow as tf
import numpy as np
import sys


def cart2polar(z):
    return tf.abs(z), tf.angle(z)


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
