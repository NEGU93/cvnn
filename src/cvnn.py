import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # Data pre-processing
    m = 500
    n = 100
    input_size = n
    output_size = 2
    total_cases = output_size*m
    train_ratio = 0.8
    # x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n)

    x_input = np.random.rand(total_cases, input_size) + 1j * np.random.rand(total_cases, input_size)
    w_real = np.random.rand(input_size, output_size) + 1j * np.random.rand(input_size, output_size)
    desired_output = np.abs(np.matmul(x_input, w_real))  # Generate my desired output

    # Separate train and test set
    x_train = x_input[:int(train_ratio * total_cases), :]
    y_train = desired_output[:int(train_ratio * total_cases), :]
    x_test = x_input[int(train_ratio * total_cases):, :]
    y_test = desired_output[int(train_ratio * total_cases):, :]

    learning_rate = 0.001
    # Network Declaration
    W = tf.Variable(tf.complex(np.random.rand(input_size, output_size),
                               np.random.rand(input_size, output_size)), name="weights")

    with tf.GradientTape(persistent=True) as gtape:
        y_out = tf.abs(tf.matmul(x_train, W), name="out")
        error = y_train - y_out
        loss = tf.reduce_mean(tf.square(tf.abs(error)), name="mse")

    epochs = 100
    for epoch in range(epochs):
        gradients = gtape.gradient(loss, [W])[0]
        W.assign_sub(learning_rate * gradients)
        print(loss)

