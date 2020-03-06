# -*- coding: utf-8 -*-
from cvnn.utils import *
import numpy as np
import sys
import os
from os.path import join
from math import sqrt
from scipy.io import loadmat
from scipy import signal
from pdb import set_trace
from abc import ABC, abstractmethod

from pylab import plot, show, axis, subplot, xlabel, ylabel, grid
from matplotlib import pyplot as plt
from scipy.linalg import eigh, cholesky
from scipy.stats import norm

"""
This files contains all functions used for pre-processing data before training which includes:
    1. Separating into train and test cases with it's corresponding cases (Utils)
    2. Create own dataset (Crate dataset)
    3. Load and save data files (Save and Load data)
"""

MARKERS = [".", "x", "s", "+", "^", "D", "_", "v", "|", "*", "H"]

"""-------------
# Dataset
-------------"""


class Dataset:
    def __init__(self, m, n, x, y, num_classes=2, ratio=0.8, savedata=False):
        """
        :param m: Number of examples per class
        :param n: Size of vector
        :param num_classes: Number of different classes to be made
        :param ratio: (float) [0, 1]. Percentage of Tran case vs Test case (ratio = #train / (#train + #test))
            Default: 0.8 (80% of the data will be used for train).
        :param savedata: (boolean) If true it will save the generated data into "./data/current/date/path/".
            Deafault: False
        """
        self.num_samples_per_class = m
        self.num_samples = n
        self.num_classes = num_classes
        self.ratio = ratio
        self.save_path = create_folder("./data/")
        self.x = x
        self.y = y
        # Generate data from x and y
        self.x_test, self.y_test = None, None               # Tests
        self.x_train, self.y_train = None, None             # Train
        self.x_train_real, self.x_test_real = None, None    # Real
        self._generate_data_from_base()
        if savedata:
            self.save_data()

    def _generate_data_from_base(self):
        """
        Generates everything (x_test, y_test, x_train, y_train, x_test_real, x_train_real) once x and y is defined.
        """
        # This must be run after each case
        self.x, self.y = randomize(self.x, self.y)
        self.x_train, self.y_train, self.x_test, self.y_test = self.separate_into_train_and_test(self.x, self.y,
                                                                                                 self.ratio)
        self.x_test_real = transform_to_real(self.x_test)
        self.x_train_real = transform_to_real(self.x_train)

    def shuffle(self):
        """
        Shuffles the train data
        """
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)

    def save_data(self):
        """
        Saves data into the specified path as a numpy array.
        """
        np.save(self.save_path / "data.npy", self.x)
        np.save(self.save_path / "labels.npy", self.y)

    def summary(self, res_str):
        """
        :return: String with the information of the dataset.
        """
        res_str += "\tNum classes: {}\n".format(self.num_classes)
        res_str += "\tSamples per class: {}\n".format(self.num_samples_per_class)
        res_str += "\tVector size: {}\n".format(self.num_samples)
        res_str += "\tTrain percentage: {}%\n".format(int(self.ratio * 100))
        res_str += "\tTest percentage: {}%\n".format(int((1 - self.ratio) * 100))
        return res_str

    def plot_data(self, overlapped=False):
        """
        Generates a figure with an example of the data
        :param overlapped: (boolean) If True it will plot all the examples in the same figure changing the color.
            Otherwise it will create a subplot with as many subplots as classes. Default: False
        :return: (fig, ax) matplolib format to be plotted.
        """

        if overlapped:
            fig, ax = plt.subplots()
            for cls in range(self.num_classes):
                for index, label in enumerate(self.y):
                    if label == cls:
                        ax.plot(np.real(self.x[index]),
                                np.imag(self.x[index]), MARKERS[cls % len(MARKERS)])
                        break
        else:
            fig, ax = plt.subplots(self.num_classes)
            for cls in range(self.num_classes):
                for index, label in enumerate(self.y):
                    if label == cls:
                        ax[cls].plot(np.real(self.x[index]),
                                     np.imag(self.x[index]), 'b.')
                        ax[cls].axis('equal')
                        ax[cls].grid(True)
                        ax[cls].set_aspect('equal', adjustable='box')
                        break
        return fig, ax

    # ---------
    # Getters
    # ---------

    def get_train_and_test(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_test(self):
        return self.x_test, self.y_test

    def get_train_test_real(self):
        return self.x_train_real, self.y_train, self.x_test_real, self.y_test

    def get_all(self):
        return self.x, self.y

    def get_categorical_labels(self):
        return self.sparse_into_categorical(self.y, self.num_classes)

    # ----------------
    # Static functions
    # ----------------

    @staticmethod
    def sparse_into_categorical(spar, num_classes=None):
        assert len(spar.shape) == 1
        spar = spar.astype(int)
        if num_classes is None:
            num_classes = max(spar) + 1  # assumes labels starts at 0
        cat = np.zeros((spar.shape[0], num_classes))
        for i, k in enumerate(spar):
            cat[i][k] = 1
        return cat

    @staticmethod
    def separate_into_train_and_test(x, y, ratio=0.8, pre_rand=True):
        """
        Separates data x with corresponding labels y into train and test set.
        :param x: data
        :param y: labels of data x
        :param ratio: value between 0 and 1.
            1 meaning all the data x will be the training set and 0 meaning all data x will be the test set.
        :param pre_rand: if True then x and y will be shuffled first (maintaining coherence between them)
        :return: tuple (x_train, y_train, x_test, y_test) of the training and test set both data and labels.
        """
        if (ratio > 1) or (ratio < 0):
            sys.exit("Error:separate_into_train_and_test: ratio should be between 0 and 1. Got value " + str(ratio))
        if pre_rand:
            x, y = randomize(x, y)
        m = np.shape(x)[0]
        x_train = x[:int(m * ratio)]
        y_train = y[:int(m * ratio)]
        x_test = x[int(m * ratio):]
        y_test = y[int(m * ratio):]
        return x_train, y_train, x_test, y_test


class GeneratorDataset(ABC, Dataset):

    def __init__(self, m, n, num_classes=2, ratio=0.8, savedata=False):
        x, y = self._generate_data(m, n, num_classes)
        super(Dataset).__init__(m, n, x, y, num_classes=num_classes, ratio=ratio, savedata=savedata)

    @abstractmethod
    def _generate_data(self, num_samples_per_class, num_samples, num_classes):
        """
        Abstract method. Must be defined to generate x and y
        """
        pass


class CorrelatedGaussianNormal(GeneratorDataset):

    def __init__(self, m, n, num_classes=2, ratio=0.8, coeff_correl_limit=0.75, debug=False, savedata=False):
        self.coeff_correl_limit = coeff_correl_limit
        self.debug = debug
        super().__init__(m, n, num_classes=num_classes, ratio=ratio, savedata=savedata)

    @staticmethod
    def _create_correlated_gaussian_point(num_samples, r=None, debug=False):
        # https: // scipy - cookbook.readthedocs.io / items / CorrelatedRandomSamples.html
        # Choice of cholesky or eigenvector method.
        method = 'cholesky'
        # method = 'eigenvectors'
        if r is None:
            # The desired covariance matrix.
            r = np.array([
                [1, 1.41],
                [1.41, 2]
            ])
        # Generate samples from three independent normally distributed random
        # variables (with mean 0 and std. dev. 1).
        x = norm.rvs(size=(2, num_samples))

        # We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
        # the Cholesky decomposition, or the we can construct `c` from the
        # eigenvectors and eigenvalues.
        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            c = cholesky(r, lower=True)
        else:
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(r)
            # Construct c, so c*c^T = r.
            c = np.dot(evecs, np.diag(np.sqrt(evals)))

        # Convert the data to correlated random variables.
        y = np.dot(c, x)
        if debug:
            plot(y[0], y[1], 'b.')
            axis('equal')
            grid(True)
            plt.gca().set_aspect('equal', adjustable='box')
            show()
        return [y[0][i] + 1j * y[1][i] for i in range(y.shape[1])]

    def _generate_data(self, num_samples_per_class, num_samples, num_classes):
        x = []
        y = []
        sigma_real = 1
        sigma_imag = 2
        for signal_class in range(num_classes):
            coeff_correl = -self.coeff_correl_limit + 2 * self.coeff_correl_limit * signal_class / (num_classes - 1)
            r = np.array([
                [sigma_real, coeff_correl * sqrt(sigma_real) * sqrt(sigma_imag)],
                [coeff_correl * sqrt(sigma_real) * sqrt(sigma_imag), sigma_imag]
            ])
            if self.debug:
                print("Class {} has coeff_correl {}".format(signal_class, coeff_correl))
                self._create_correlated_gaussian_point(num_samples, r, True)
            y.extend(signal_class * np.ones(num_samples_per_class))
            for _ in range(num_samples_per_class):
                x.append(self._create_correlated_gaussian_point(num_samples, r, debug=False))
        return np.array(x), np.array(y)

    def summary(self, res_str=None):
        res_str = "Correlated Gaussian Noise\n"
        res_str += "\tPearson correlation coefficient max {}\n".format(self.coeff_correl_limit)
        return super().summary(res_str)


class GaussianNoise(GeneratorDataset):

    def __init__(self, m, n, num_classes=2, ratio=0.8, savedata=False, function='hilbert'):
        noise_gen_dispatcher = {
            'non_correlated': self._create_non_correlated_gaussian_noise,
            'hilbert': self._create_hilbert_gaussian_noise
        }
        try:
            self.function = noise_gen_dispatcher[function]
        except KeyError:
            sys.exit("GaussianNoise: Unknown type of noise" + str(function))
        super().__init__(m, n, num_classes=num_classes, ratio=ratio, savedata=savedata)

    def _generate_data(self, num_samples_per_class, num_samples, num_classes):
        x = np.empty((num_classes * num_samples_per_class, num_samples)) \
                 + 1j * np.empty((num_classes * num_samples_per_class, num_samples))
        # I am using zeros instead of empty because although counter intuitive it seams it works faster:
        # https://stackoverflow.com/questions/55145592/performance-of-np-empty-np-zeros-and-np-ones
        # DEBUNKED? https://stackoverflow.com/questions/52262147/speed-of-np-empty-vs-np-zeros?
        # Initialize all at 0 to later put a 1 on the corresponding place
        # TODO: generate to zero the other parameters as well
        y = np.zeros((num_classes * num_samples_per_class, num_classes))

        for k in range(num_classes):
            mu = int(100 * np.random.rand())
            sigma = 15 * np.random.rand()
            print("Class " + str(k) + ": mu = " + str(mu) + "; sigma = " + str(sigma))
            x[k * num_samples_per_class:(k + 1) * num_samples_per_class, :] = self.function(num_samples_per_class,
                                                                                            num_samples, mu, sigma)
            y[k * num_samples_per_class:(k + 1) * num_samples_per_class, k] = 1
        return normalize(x), y

    @staticmethod
    def _create_non_correlated_gaussian_noise(num_samples_per_class, num_samples, mu, sigma):
        """
        Creates a numpy matrix of size mxn with random gaussian distribution of mean mu and variance sigma
        """
        return (np.random.normal(mu, sigma, (num_samples_per_class, num_samples)) +
                1j * np.random.normal(mu, sigma, (num_samples_per_class, num_samples))) / sqrt(2)

    @staticmethod
    def _create_hilbert_gaussian_noise(num_samples_per_class, num_samples, mu, sigma):
        x_real = np.random.normal(mu, sigma, (num_samples_per_class, num_samples))
        return signal.hilbert(x_real)

    def summary(self, res_str=None):
        res_str = "Gaussian {} Noise\n".format(str(self.function).replace('_', ' '))
        return super().summary(res_str)


"""-----------------
# Save and Load data
-----------------"""


def save_npy_array(array_name, array):
    np.save("./data/" + array_name + ".npy", array)


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
    os.makedirs("../data", exist_ok=True)
    return np.savez("../data/" + array_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


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
        sys.exit("Cvnn::load_dataset: The file could not be found")  # TODO: check if better just throw a warning


def load_matlab_matrices(fname="data_cnn1dT.mat", path="/media/barrachina/data/gilles_data/"):
    mat_fname = join(path, fname)
    mat = loadmat(mat_fname)
    return mat


"""----------------
# Testing Functions
----------------"""


def test_save_load():
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


def create_subplots_of_graph():
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 5
    n = 100
    num_classes = 2
    coefs = [0.1, 0.4, 0.75, 0.999]
    overlaped = True
    rows = 2
    if overlaped:
        rows = 1
    fig, axs = plt.subplots(rows, len(coefs), sharex=True, sharey=True)
    # , gridspec_kw={'hspace': 0, 'wspace': 0})
    for i, coef in enumerate(coefs):
        dataset = CorrelatedGaussianNormal(m, n, num_classes=num_classes, debug=False, coeff_correl_limit=coef)
        x, y = dataset.get_all()
        for r in range(rows):
            for cls in range(num_classes):
                for index, label in enumerate(y):
                    if label == cls:
                        if overlaped:
                            axs[i].plot(np.real(x[index]), np.imag(x[index]), MARKERS[cls % len(MARKERS)])
                            axs[i].axis('equal')
                            axs[i].grid(True)
                            axs[i].set_aspect('equal', adjustable='box')
                            break
                        else:
                            axs[r, i].plot(np.real(x[index]), np.imag(x[index]), 'b.')
                            axs[r, i].axis('equal')
                            axs[r, i].grid(True)
                            axs[r, i].set_aspect('equal', adjustable='box')
                            break
    if overlaped:
        for ax, coef in zip(axs, coefs):
            ax.set_title("coef abs: {}".format(coef))
        for cls, ax in enumerate(axs):
            ax.set_ylabel("class {}".format(int(cls)), size='large')
    else:
        for ax, coef in zip(axs[0], coefs):
            ax.set_title("coef abs: {}".format(coef))
        for cls, ax in enumerate(axs[:, 0]):
            ax.set_ylabel("class {}".format(int(cls)), size='large')
    fig.show()
    # create_correlated_gaussian_noise(n, debug=True)
    # set_trace()


if __name__ == "__main__":
    # create_subplots_of_graph()
    m = 5
    n = 100
    num_classes = 2
    dataset = CorrelatedGaussianNormal(m, n, num_classes=num_classes)
    set_trace()

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.1.5'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
