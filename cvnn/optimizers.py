from abc import ABC, abstractmethod
import tensorflow as tf
from cvnn import logger
import sys
from typing import Union
from cvnn.layers import t_layers_shape


class Optimizer(ABC):
    def __init__(self):
        pass

    def compile(self, shape: t_layers_shape) -> None:
        pass

    def optimize(self, variables, gradients):
        pass

    def summary(self) -> str:
        """
        :returns: A one line short string with the description of the optimizer
        """
        pass

    def __deepcopy__(self, memodict=None):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, name: str = 'SGD'):
        """
        Gradient descent (with momentum) optimizer.

        :param learning_rate: The learning rate. Defaults to 0.001.
        :param momentum: float hyperparameter between [0, 1) that accelerates gradient descent in the relevant
                        direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
        :param name: Optional name for the operations created when applying gradients. Defaults to "Adam".
        """
        self.name = name
        self.learning_rate = learning_rate
        if momentum > 1 or momentum < 0:
            logger.error("momentum must be between 1 and 0. {} was given".format(momentum))
            sys.exit(-1)
        self.momentum = momentum
        self.velocity = []
        self.first_time = True
        super().__init__()

    def __deepcopy__(self, memodict={}):
        if memodict is None:
            memodict = {}
        return SGD(learning_rate=self.learning_rate, momentum=self.momentum, name=self.name)

    def summary(self) -> str:
        return "SDG optimizer " + self.name + \
               ": learning rate = " + str(self.learning_rate) + \
               "; momentum = " + str(self.momentum) + "\n"

    def compile(self, shape: t_layers_shape) -> None:
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
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, name="RMSprop"):
        """
        Optimizer that implements the RMSprop algorithm.
        Reference: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

        The gist of RMSprop is to:
            - Maintain a moving (discounted) average of the square of gradients
            - Divide the gradient by the root of this average
            - This implementation of RMSprop uses plain momentum, not Nesterov momentum.
        The centered version additionally maintains a moving average of the gradients, and uses that average to estimate the variance.

        :param learning_rate: The learning rate. Defaults to 0.001.
        :param rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
        :param momentum: The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        :param epsilon: A small constant for numerical stability. Default 1e-07.
        :param name: Optional name for the operations created when applying gradients. Defaults to "Adam".
        """
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

    def __deepcopy__(self, memodict={}):
        if memodict is None:
            memodict = {}
        return RMSprop(learning_rate=self.learning_rate, rho=self.rho, momentum=self.momentum, epsilon=self.epsilon,
                       name=self.name)

    def summary(self) -> str:
        return "RMSprop optimizer " + self.name + \
               ": learning rate = " + str(self.learning_rate) + " rho = " + str(self.rho) + \
               "; momentum = " + str(self.momentum) + "; epsilon = " + str(self.epsilon) + "\n"

    def compile(self, shape: t_layers_shape) -> None:
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
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-07, name="Adam"):
        """
        Optimizer that implements the Adam algorithm.
        Reference: https://arxiv.org/abs/1412.6980
        Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of
                        first-order and second-order moments.

        :param learning_rate: The learning rate. Defaults to 0.001.
        :param beta_1: The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        :param beta_2: The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        :param epsilon: A small constant for numerical stability. Default 1e-07.
        :param name: Optional name for the operations created when applying gradients. Defaults to "Adam".
        """
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

    def __deepcopy__(self, memodict={}):
        if memodict is None:
            memodict = {}
        return Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
                    name=self.name)

    def summary(self) -> str:
        return "RMSprop optimizer " + self.name + \
               ": learning rate = " + str(self.learning_rate) + " beta_1 = " + str(self.beta_1) + \
               "; beta_2 = " + str(self.beta_2) + "; epsilon = " + str(self.epsilon) + "\n"

    def compile(self, shape: t_layers_shape) -> None:
        for layer in shape:
            for elem in layer.trainable_variables():
                self.vdw.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))
                self.sdw.append(tf.Variable(tf.zeros(elem.shape, dtype=layer.get_input_dtype())))

    def optimize(self, variables, gradients):
        with tf.name_scope(self.name):
            for i, val in enumerate(variables):
                self.vdw[i].assign(tf.add(
                    tf.scalar_mul(self.beta_1, self.vdw[i]),
                    tf.scalar_mul(1 - self.beta_1, gradients[i])))
                self.sdw[i].assign(tf.add(
                    tf.scalar_mul(self.beta_2, self.sdw[i]),
                    tf.scalar_mul(1 - self.beta_2, tf.math.square(gradients[i]))))
                vdw_corr = tf.math.divide(self.vdw[i], tf.math.pow(1 - self.beta_1, self.iter))
                sdw_corr = tf.math.divide(self.sdw[i], tf.math.pow(1 - self.beta_2, self.iter))
                val.assign(val - self.learning_rate * vdw_corr / (tf.math.sqrt(sdw_corr) + self.epsilon))
            self.iter += 1


t_optimizer = Union[str, Optimizer]


def get_optimizer(optimizer: t_optimizer) -> Optimizer:
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


opt_dispatcher = {
    'sgd': SGD(),
    'rmsprop': RMSprop(),
    'adam': Adam(),
}
