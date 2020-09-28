from abc import ABC, abstractmethod
import tensorflow as tf
from cvnn import logger
import numpy as np
import sys
from pdb import set_trace


class Optimizer(ABC):
    def __init__(self):
        pass

    # def init_velocity(self, shape):
    #    pass

    def optimize(self, variables, gradients):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, name: str = 'SGD'):
        self.name = name
        self.learning_rate = learning_rate
        if self.momentum > 1 or self.momentum < 0:
            logger.error("momentum must be between 1 and 0. {} was given".format(momentum))
            sys.exit(-1)
        self.momentum = momentum
        self.velocity = []
        self.first_time = True
        super().__init__()

    """
    def init_velocity(self, shape):
        for layer in shape:
            for elem in layer.trainable_variables():
                self.velocity.append(tf.zeros(elem.shape, dtype=layer.get_input_dtype()))"""

    def optimize(self, variables, gradients):
        with tf.name_scope(self.name):
            for i, val in enumerate(variables):
                if self.first_time:
                    self.velocity.append((1-self.momentum) * gradients[i])
                else:
                    self.velocity[i] = (1 - self.momentum) * gradients[i]
                    # self.velocity[i] = self.momentum*self.velocity[i] + (1 - self.momentum) * gradients[i]
                val.assign(val - self.learning_rate * self.velocity[i])
            self.first_time = False
        return variables


