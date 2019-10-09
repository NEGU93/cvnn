import tensorflow as tf
import numpy as np

# Create data:
input_size = 30
output_size = 1
total_cases = 10000
train_ratio = 0.8

x_input = np.random.rand(total_cases, input_size) + 1j * np.random.rand(total_cases, input_size)
w_real = np.random.rand(input_size, output_size) + 1j * np.random.rand(input_size, output_size)
desired_output = np.matmul(x_input, w_real)  # Generate my desired output

# Separate train and test set
x_train = x_input[:int(train_ratio * total_cases), :]
y_train = desired_output[:int(train_ratio * total_cases), :]
x_test = x_input[int(train_ratio * total_cases):, :]
y_test = desired_output[int(train_ratio * total_cases):, :]

# Hyper-parameters
epochs = 100  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 50  # Display results frequency
learning_rate = 0.001  # The optimization initial learning rate


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


# Reset latest graph
tf.reset_default_graph()

# Define placeholders
X = tf.placeholder(tf.complex64, shape=[None, input_size], name='X')
y = tf.placeholder(tf.complex64, shape=[None, output_size], name='Y')

# Create weight matrix initialized randomely from N~(0, 0.01)
W = tf.Variable(tf.complex(np.random.rand(input_size, output_size).astype(np.float32),
                           np.random.rand(input_size, output_size).astype(np.float32)), name="W")

y_out = tf.matmul(X, W, name="out")

# Define Graph
error = y - y_out
mse = tf.reduce_mean(tf.square(tf.abs(error)), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# gradients = tf.gradients(mse, [weights])[0]
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)

    for epoch in range(epochs):
        # Randomly shuffle the training data at the beginning of each epoch
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
            # Run optimization op (backprop)
            feed_dict_batch = {X: x_batch, y: y_batch}
            sess.run(training_op, feed_dict=feed_dict_batch)
            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch = sess.run(mse, feed_dict=feed_dict_batch)
                print("iter {0:3d}:\t Loss={1:.2f}".format(iteration, loss_batch))

    # Run validation after every epoch
    feed_dict_valid = {X: x_test, y: y_test}
    loss_valid = sess.run(mse, feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.4f}".format(epoch + 1, loss_valid))
    print('---------------------------------------------------------')

    print(np.mean(np.abs(y_test - np.matmul(x_test, W.eval())) ** 2))
    print(y_test[:3])
    print(y.eval(feed_dict=feed_dict_valid)[:3])