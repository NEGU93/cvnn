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
epochs = 200  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 40  # Display results frequency
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
tf.compat.v1.reset_default_graph()

# Define placeholders
X = tf.compat.v1.placeholder(tf.complex64, shape=[None, input_size], name='X')
y = tf.compat.v1.placeholder(tf.complex64, shape=[None, output_size], name='Y')

# Create weight matrix initialized randomely from N~(0, 0.01)
W = tf.Variable(tf.complex(np.random.rand(input_size, output_size).astype(np.float32),
                           np.random.rand(input_size, output_size).astype(np.float32)), name="W")

y_out = tf.matmul(X, W, name="out")

# Define Graph
error = y_out - y
mse = tf.reduce_mean(input_tensor=tf.square(tf.abs(error)), name="mse")

gradients = 2 / batch_size * tf.matmul(tf.transpose(a=tf.math.conj(X)), error)
autodiff = tf.gradients(ys=mse, xs=[W])[0]

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    iteration = 0
    x_train, y_train = randomize(x_train, y_train)

    start = iteration * batch_size
    end = (iteration + 1) * batch_size
    x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
    # Run optimization op (backprop)
    feed_dict_batch = {X: x_batch, y: y_batch}

    print("Manual Gradients: \n" + str(gradients.eval(feed_dict=feed_dict_batch)[:3]))
    print("Autodiff Gradients: \n" + str(autodiff.eval(feed_dict=feed_dict_batch)[:3]))