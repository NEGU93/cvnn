import tensorflow as tf

"""
# Loss functions
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

All loss functions will be declared here. 
Loss functions will be type agnostic, meaning it can be with either complex or real value outputs and labels.
"""

"""---------
# REGRESSION
---------"""


def mean_square(y_out, y):
    """
    Mean Squared Error, or L2 loss.
    :return:
    """
    with tf.compat.v1.name_scope("loss") as scope:
        error = y - y_out
        return tf.reduce_mean(input_tensor=tf.square(tf.abs(error)), name="loss")


"""-------------
# CLASSIFICATION
-------------"""


def categorical_crossentropy(y_out, y):
    """
    https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
    :return: -y*log(y_out)-(1-y)*log(1-y_out) where:
        log - the natural log
        y - binary indicator (0 or 1), it will be all 0's but one (according to the corresponding class)
        y_out - predicted probability observation the class
    """
    with tf.compat.v1.name_scope("loss_scope") as scope:
        y1_error = tf.math.multiply(-y, tf.math.log(y_out))  # Error for y = 1
        y0_error = tf.math.multiply(1 - y, tf.math.log(1 - y_out))  # Error for y = 0
        error = tf.math.subtract(y1_error, y0_error)
        return tf.reduce_mean(input_tensor=error, name="loss")


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '1.0.0'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'