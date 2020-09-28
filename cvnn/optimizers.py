from abc import ABC, abstractmethod
import tensorflow as tf


class Optimizer(ABC):
    def __init__(self):
        pass

    def optimize(self, variables, gradients):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, name: str = 'SGD'):
        self.name = name
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__()

    def optimize(self, variables, gradients):
        with tf.name_scope(self.name):
            for i, val in enumerate(variables):
                val.assign(val - self.learning_rate * gradients[i])
        return variables


