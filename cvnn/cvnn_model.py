import os
import sys
import logging
import numpy as np
import tensorflow as tf
import cvnn
import cvnn.layers as layers
import cvnn.data_processing as dp
from cvnn.utils import normalize, randomize, get_next_batch
from pdb import set_trace

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class OutputOpts:
    def __init__(self, tensorboard, verbose, display_freq=100):
        self.tensorboard = tensorboard
        self.verbose = verbose
        self.display_freq = display_freq


class CvnnModel:
    """-------------------------
    # Constructor and Stuff
    -------------------------"""

    def __init__(self, name, shape, loss_fun, verbose=True, tensorboard=False, display_freq=100):
        # logging.getLogger('tensorflow').disabled = True
        logger = logging.getLogger(cvnn.__name__)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        self.logger = logger

        self.name = name
        self.shape = shape
        self.variables = []
        self.loss_fun = loss_fun
        self.output_options = OutputOpts(tensorboard, verbose, display_freq)

    def __call__(self, x):
        out = x
        self.variables.clear()
        for i in range(len(self.shape)):  # Apply all the layers
            out, variable = self.shape[i].apply_layer(out)
            self.variables.extend(variable)
            # set_trace()
        # self.variables = variables
        y_out = out     # tf.compat.v1.identity(out, name="y_out")
        return y_out

    def _apply_loss(self, y_true, y_pred):
        # TODO: don't like the fact that I have to give self to this and not to apply_activation
        if callable(self.loss_fun):
            if self.loss_fun.__module__ != 'tensorflow.python.keras.losses':
                self.logger.error("Unknown loss function.\n\t "
                                  "Can only use losses declared on tensorflow.python.keras.losses")
        return tf.reduce_mean(input_tensor=self.loss_fun(y_true, y_pred), name=self.loss_fun.__name__)

    """-------------------------
    # Predict models and results
    -------------------------"""

    def predict(self, x):
        y_out = self.__call__(x)
        return tf.math.argmax(y_out, 1)

    def evaluate_loss(self, x_test, y_true):
        return self._apply_loss(y_true, self.__call__(x_test)).numpy()

    def evaluate_accuracy(self, x_test, y_true):
        y_pred = self.predict(x_test)
        y_labels = tf.math.argmax(y_true, 1)
        return tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(y_pred, y_labels), tf.float64)).numpy()

    def evaluate(self, x_test, y_true):
        return self.evaluate_loss(x_test, y_true), self.evaluate_accuracy(x_test, y_true)

    """-----------------------
    #          Train 
    -----------------------"""

    def _single_train(self, x_train_batch, y_train_batch, learning_rate):
        with tf.GradientTape() as t:
            current_loss = self._apply_loss(y_train_batch, self.__call__(x_train_batch))    # Compute loss
        # print(len(self.variables))
        gradients = t.gradient(current_loss, self.variables)                                # Compute gradients
        set_trace()
        assert all(g is not None for g in gradients)
        # set_trace()
        for i, lay in enumerate(self.shape):
            # tf.compat.v1.assign(var, var - learning_rate*gradients[i])  # Change values
            lay.update_weights(learning_rate*gradients[2*i], learning_rate*gradients[2*i+1])

    def fit(self, x_train, y_train, learning_rate=0.01, epochs=10, batch_size=100, normal=True):
        if np.shape(x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
            self.logger.error("Batch size was bigger than total amount of examples")
        if normal:
            x_train = normalize(x_train)  # TODO: This normalize could be a bit different for each and be bad.

        num_tr_iter = int(len(y_train) / batch_size)  # Number of training iterations in each epoch
        if self.output_options.verbose:
            print("Starting training...")
        for epoch in range(epochs):
            # Randomly shuffle the training data at the beginning of each epoch
            x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                # Get the batch
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                # Run optimization op (backpropagation)
                if self.output_options.verbose and \
                        (epoch * batch_size + iteration) % self.output_options.display_freq == 0:
                    current_loss, current_acc = self.evaluate(x_batch, y_batch)
                    print("Epoch: {0}/{1}; batch {2}/{3}; loss: {4:.4f} accuracy: {5:.2f} %"
                          .format(epoch, epochs, iteration, num_tr_iter, current_loss, current_acc*100))
                self._single_train(x_batch, y_batch, learning_rate)


if __name__ == '__main__':
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 1000
    n = 100
    num_classes = 2
    x_train, y_train, x_test, y_test = dp.get_gaussian_noise(m, n, num_classes, 'hilbert')

    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    shape = [layers.Dense(input_size=input_size, output_size=hidden_size, activation='cart_sigmoid',
                          input_dtype=np.complex64, output_dtype=np.complex64),
             layers.Dense(input_size=hidden_size, output_size=output_size, activation='cart_softmax_real',
                          input_dtype=np.complex64, output_dtype=np.float32)]
    model = CvnnModel("Testing CVNN", shape, tf.keras.losses.categorical_crossentropy)
    print(model.evaluate(x_test.astype(np.complex64), y_test))
    model.fit(x_train.astype(np.complex64), y_train, learning_rate=0.1, batch_size=100, epochs=10, normal=True)
    print(model.evaluate(x_test.astype(np.complex64), y_test))


