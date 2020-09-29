from abc import ABC, abstractmethod
import tensorflow as tf
from cvnn import logger
import numpy as np
import sys
from pdb import set_trace


def get_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer
    elif isinstance(optimizer, str):
        try:
            # TODO: For the moment is not possible to give parameters to constructors
            return opt_dispatcher[optimizer.lower()]
        except KeyError:
            logger.warning(str(optimizer) + " is not a known optimizer. Known optimizers:" +
                           s for s in opt_dispatcher.keys())
            sys.exit(-1)


class Optimizer(ABC):
    def __init__(self):
        pass

    def compile(self, shape):
        pass

    def optimize(self, variables, gradients):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, name: str = 'SGD'):
        self.name = name
        self.learning_rate = learning_rate
        if momentum > 1 or momentum < 0:
            logger.error("momentum must be between 1 and 0. {} was given".format(momentum))
            sys.exit(-1)
        self.momentum = momentum
        self.velocity = []
        self.first_time = True
        super().__init__()

    def compile(self, shape):
        for layer in shape:
            for elem in layer.trainable_variables():
                self.velocity.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))

    def optimize(self, variables, gradients):
        with tf.name_scope(self.name):
            for i, val in enumerate(variables):
                if self.first_time:
                    self.velocity.append(tf.Variable((1-self.momentum) * gradients[i]))
                else:
                    self.velocity[i].assign(self.momentum*self.velocity[i] + (1 - self.momentum) * gradients[i])
                val.assign(val - self.learning_rate * self.velocity[i])
            self.first_time = False


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, name="Adam"):
        self.name = name
        self.learning_rate = learning_rate
        if rho > 1 or rho < 0:
            logger.error("rho must be between 1 and 0. {} was given".format(rho))
            sys.exit(-1)
        if rho > 1 or rho < 0:
            logger.error("momentum must be between 1 and 0. {} was given".format(momentum))
            sys.exit(-1)
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.vdw = []
        self.sdw = []
        super().__init__()

    def compile(self, shape):
        for layer in shape:
            for elem in layer.trainable_variables():
                self.vdw.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))
                self.sdw.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))

    def optimize(self, variables, gradients):
        with tf.name_scope(self.name):
            for i, val in enumerate(variables):
                self.vdw[i].assign(self.momentum * self.vdw[i] + (1 - self.momentum) * gradients[i])
                self.sdw[i].assign(self.rho * self.sdw[i] + (1 - self.rho) * tf.math.square(gradients[i]))
                val.assign(val - self.learning_rate * self.vdw[i] / tf.math.sqrt(self.sdw[i] + self.epsilon))


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam"):
        self.name = name
        self.learning_rate = learning_rate
        if beta_1 >= 1 or beta_1 < 0:
            logger.error("beta_1 must be between [0, 1). {} was given".format(beta_1))
            sys.exit(-1)
        if beta_2 >= 1 or beta_2 < 0:
            logger.error("beta_2 must be between [0, 1). {} was given".format(beta_2))
            sys.exit(-1)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.vdw = []
        self.sdw = []
        self.iter = 1
        super().__init__()

    def compile(self, shape):
        for layer in shape:
            for elem in layer.trainable_variables():
                self.vdw.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))
                self.sdw.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))

    def optimize(self, variables, gradients):
        with tf.name_scope(self.name):
            for i, val in enumerate(variables):
                self.vdw[i].assign(self.beta_1 * self.vdw[i] + (1 - self.beta_1) * gradients[i])
                self.sdw[i].assign(self.beta_2 * self.sdw[i] + (1 - self.beta_2) * tf.math.square(gradients[i]))
                vdw_corr = self.vdw[i] / (1 - self.beta_1**self.iter)
                sdw_corr = self.sdw[i] / (1 - self.beta_2**self.iter)
                val.assign(val - self.learning_rate * vdw_corr / (tf.math.sqrt(sdw_corr) + self.epsilon))
            self.iter += 1


opt_dispatcher = {
    'sgd': SGD(),
    'rmsprop': RMSprop(),
    'adam': Adam(),
}
