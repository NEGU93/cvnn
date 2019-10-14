import numpy as np
import sys
import os

"""----
# Utils
----"""


def randomize(x, y):
    """
    Randomizes the order of data samples and their corresponding labels
    :param x: data
    :param y: data labels
    :return: Tuple of (shuffled_x, shuffled_y) maintaining coherence of elements labels
    """
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    """
    Get next batch from x and y using start and end
    :param x: data
    :param y: data labels
    :param start: starting index of the batch to be returned
    :param end: end index of the batch to be returned (not including)
    :return: tuple (x, y) of the selected batch
    """
    if start < 0:
        sys.exit("Error:get_next_batch(): start parameter cannot be negative")
    if start > end:     # What will happen if not? Should I leave this case anyway and just give a warning?
        sys.exit("Error:get_next_batch(): end should be higher than start")
    # TODO: Check end < len(x)
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def transform_to_real(x_complex):
    """
    :param x_complex: Complex-valued matrix of size mxn
    :return: real-valued matrix of size mx(2*n) unwrapping the real and imag part of the complex-valued input matrix
    """
    m = np.shape(x_complex)[0]
    n = np.shape(x_complex)[1]
    x_real = np.ones((m, 2*n))
    x_real[:, :n] = np.real(x_complex)
    x_real[:, n:] = np.imag(x_complex)
    return x_real


def separate_into_train_and_test(x, y, ratio=0.8):
    """
    Separates data x with corresponding labels y into train and test set.
    :param x: data
    :param y: labels of data x
    :param ratio: value between 0 and 1.
        1 meaning all the data x will be the training set and 0 meaning all data x will be the test set.
    :return: tuple (x_train, y_train, x_test, y_test) of the training and test set both data and labels.
    """
    if (ratio > 1) or (ratio < 0):
        sys.exit("Error:separate_into_train_and_test: ratio should be between 0 and 1. Got value " + str(ratio))
    m = np.shape(x)[0]
    x_train = x[:int(m*ratio)]
    y_train = y[:int(m*ratio)]
    x_test = x[int(m*ratio):]
    y_test = y[int(m*ratio):]
    return x_train, y_train, x_test, y_test


"""-------------
# Create dataset
-------------"""


def _create_data(m, n, mu, sigma):
    """
    Creates a numpy matrix of size mxn with random gaussian distribution of mean mu and variance sigma
    """
    x = mu + 1j*mu + sigma * (np.random.rand(m, n) + 1j * np.random.rand(m, n))
    return x


def _create_non_correlated_gaussian_noise(m, n, num_classes=2):
    """

    :param m: Number of examples per class
    :param n: Size of vector
    :param num_classes: Number of different classes to be made
    :return: tuple of a (num_classes*m)xn matrix with data and labels regarding it class.
    """
    x = np.empty((num_classes*m, n)) + 1j*np.ones((num_classes*m, n))
    # I am using zeros instead of empty because although counter intuitive it seams it works faster:
    # https://stackoverflow.com/questions/55145592/performance-of-np-empty-np-zeros-and-np-ones
    # DEBUNKED? https://stackoverflow.com/questions/52262147/speed-of-np-empty-vs-np-zeros?
    y = np.zeros((num_classes*m, num_classes))      # Initialize all at 0 to later put a 1 on the corresponding place
    for k in range(num_classes):
        mu = int(100*np.random.rand())
        sigma = int(10*np.random.rand())
        x[k*m:(k+1)*m, :] = _create_data(m, n, mu, sigma)
        y[k*m:(k+1)*m, k] = 1

    # import pdb; pdb.set_trace()
    return x, y


def get_non_correlated_gaussian_noise(m, n, num_classes=2):
    x, y = _create_non_correlated_gaussian_noise(m, n, num_classes)
    x, y = randomize(x, y)
    return separate_into_train_and_test(x, y)


"""-----------------
# Save and Load data
-----------------"""


def save_npy_array(array_name, array):
    np.save("../data/"+array_name+".npy", array)


def save_dataset(array_name, x_train, y_train, x_test, y_test):
    """
    Saves in a single .npz file the test and training set with corresponding labels
    :param array_name: Name of the array to be saved into data/ folder.
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: None
    """
    if not os.path.exists("../data"):
        os.makedirs("../data")
    return np.savez("../data/"+array_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def load_dataset(array_name):
    """
    Gets all x_train, y_train, x_test, y_test from a previously saved .npz file with save_dataset function.
    :param array_name: name of the file saved in '../data' with .npz termination
    :return: tuple (x_train, y_train, x_test, y_test)
    """
    try:
        # print(os.listdir("../data"))
        npzfile = np.load("../data/" + array_name + ".npz")
        # print(npzfile.files)
        return npzfile['x_train'], npzfile['y_train'], npzfile['x_test'], npzfile['y_test']
    except FileNotFoundError:
        sys.exit("Cvnn::load_dataset: The file could not be found")     # TODO: check if better just throw a warning


if __name__ == "__main__":
    # Data pre-processing
    m = 5000
    n = 30
    input_size = n
    output_size = 1
    total_cases = 2 * m
    train_ratio = 0.8
    # x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n)

    x_input = np.random.rand(total_cases, input_size) + 1j * np.random.rand(total_cases, input_size)
    w_real = np.random.rand(input_size, output_size) + 1j * np.random.rand(input_size, output_size)
    desired_output = np.matmul(x_input, w_real)  # Generate my desired output

    # Separate train and test set
    x_train = x_input[:int(train_ratio * total_cases), :]
    y_train = desired_output[:int(train_ratio * total_cases), :]
    x_test = x_input[int(train_ratio * total_cases):, :]
    y_test = desired_output[int(train_ratio * total_cases):, :]

    save_dataset("linear_output", x_train, y_train, x_test, y_test)
    x_loaded_train, y_loaded_train, x_loaded_test, y_loaded_test = load_dataset("linear_output")

    if np.all(x_train == x_loaded_train):
        if np.all(y_train == y_loaded_train):
            if np.all(x_test == x_loaded_test):
                if np.all(y_test == y_loaded_test):
                    print("All good!")
