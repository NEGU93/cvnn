import numpy as np
import tensorflow as tf
import cvnn.layers as layers
import cvnn.data_processing as dp


class CvnnModel:
    """-------------------------
    # Constructor and Destructor
    -------------------------"""

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.variables = []

    def __call__(self, x):
        out = x
        for i in range(len(self.shape)):  # Apply all the layers
            out, variable = self.shape[i].apply_layer(out)
            self.variables.extend(variable)
        y_out = tf.compat.v1.identity(out, name="y_out")
        return y_out


def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.square(tf.abs(predicted_y - target_y)))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    gradients = t.gradient(current_loss, model.variables)
    for i, var in enumerate(model.variables):
        tf.compat.v1.assign(var, var - learning_rate * gradients[i])
    # model.W.assign_sub(learning_rate * dW)
    # model.b.assign_sub(learning_rate * db)


if __name__ == '__main__':
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 10000
    n = 100
    num_classes = 5
    x_train, y_train, x_test, y_test = dp.get_gaussian_noise(m, n, num_classes, 'hilbert')

    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]

    shape = [layers.Dense(input_size=input_size, output_size=hidden_size, activation='cart_sigmoid',
                          input_dtype=np.complex64, output_dtype=np.complex64),
             layers.Dense(input_size=hidden_size, output_size=output_size, activation='cart_softmax_real',
                          input_dtype=np.complex64, output_dtype=np.complex64)]
    model = CvnnModel("Testing CVNN", shape)

    epochs = range(10)
    for epoch in epochs:
        current_loss = loss(model(x_train), y_train)
        train(model, x_train, y_train, learning_rate=0.001)


