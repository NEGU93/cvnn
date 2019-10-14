import tensorflow as tf
import numpy as np

def cart2polar(z):
    return tf.abs(z), tf.angle(z)


def polar2cart(rho, angle):
    return rho * np.exp(1j*angle)
