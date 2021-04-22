# -*- coding: utf-8 -*-
from cvnn.utils import *
import numpy as np
import sys
from math import sqrt
from scipy import signal
from pdb import set_trace
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import plotly
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
import tikzplotlib

MARKERS = ["*", "s", "x", "+", "^", "D", "_", "v", "|", ".", "H"]
COLORS = list(mcolors.BASE_COLORS)
logger = logging.getLogger(cvnn.__name__)


# =======
# Dataset
# =======


class Dataset:
    """
    This class is used to centralize all dataset management.
    """

    def __init__(self, x, y, num_classes=None, ratio=0.8, savedata=False, batch_size=None,
                 categorical=True, debug=False, shuffle=False, dataset_name=""):
        """
        :param x: Data
        :param y: Labels/outputs
        :param num_classes: Number of different classes to be made
        :param ratio: (float) [0, 1]. Percentage of Tran case vs Test case (ratio = #train / (#train + #test))
            Default: 0.8 (80% of the data will be used for train).
        :param savedata: (boolean) If true it will save the generated data into "./data/current/date/path/".
            Default: False
        """
        self.dataset_name = dataset_name
        self.random_shuffle = shuffle
        x = np.array(x)
        y = np.array(y)
        if x.dtype == np.complex128:  # Do this cast not to have warning messages when fit
            x = x.astype(np.complex64)
        elif x.dtype == np.float64:
            x = x.astype(np.float32)
        self.x = x
        self.y = y.astype(np.float32)
        self.categorical = categorical  # TODO: know it automatically as done in other functions
        if categorical:
            self.y = self.sparse_into_categorical(self.y)
        if num_classes is None:
            self.num_classes = self._deduce_num_classes()  # This is only used for plotting the data example
        else:
            self.num_classes = num_classes
        self.ratio = ratio
        self.save_path = "./data/"
        # Generate data from x and y
        self.x_test, self.y_test = None, None  # Tests
        self.x_train, self.y_train = None, None  # Train
        self._generate_data_from_base()
        if savedata:
            self.save_data()
        # Parameters used with the fit method
        self._iteration = 0
        if batch_size is None:
            self.batch_size = self.x_train.shape[0]  # Don't use batches at all
        else:
            if np.shape(self.x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
                logger.error("Batch size was bigger than total amount of examples")
                sys.exit(-1)
            self.batch_size = batch_size
        if debug:
            self.plot_data(overlapped=True, showfig=True, save_path=None)

    def get_next_batch(self):
        num_tr_iter = int(self.x_train.shape[0] / self.batch_size)  # Number of training iterations in each epoch
        if not self._iteration < num_tr_iter:
            logger.error("I did more calls to this function that planned")
            sys.exit(-1)
        # Get the next batch
        start = self._iteration * self.batch_size
        end = (self._iteration + 1) * self.batch_size
        self._iteration += 1

        return self._get_next_batch(self.x_train, self.y_train, start, end)

    def _deduce_num_classes(self):
        """
        Tries to deduce the total amount of classes present in the network.
        ATTENTION: This method will obviously fail for regression data.
        """
        # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        if len(self.y.shape) == 1:  # Sparse labels
            num_samples = max(self.y.astype(int)) - min(self.y.astype(int)) + 1
        else:  # Categorical labels
            num_samples = self.y.shape[1]
        return num_samples

    def _generate_data_from_base(self):
        """
        Generates everything (x_test, y_test, x_train, y_train, x_real) once x and y is defined.
        """
        self.x_train, self.y_train, self.x_test, self.y_test = self.separate_into_train_and_test(self.x, self.y,
                                                                                                 self.ratio,
                                                                                                 pre_rand=self.random_shuffle)

    def shuffle(self):
        """
        Shuffles the train data and reset iteration counter
        """
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)
        self._iteration = 0

    def save_data(self, save_path=None):
        """
        Saves data into the specified path as a numpy array.
        """
        if save_path is None:
            save_path = create_folder(self.save_path)
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = Path(save_path)
        if os.path.exists(save_path):
            np.save(save_path / "data.npy", self.x)
            np.save(save_path / "labels.npy", self.y)
            # Save also an image of the example
            self.plot_data(overlapped=True, showfig=False, save_path=save_path)
        else:
            logger.error("Path {} does not exist".format(save_path))

    def summary(self, res_str=None):
        """
        :return: String with the information of the dataset.
        """
        if res_str is None:
            res_str = self.dataset_name
        res_str += "\tNum classes: {}\n".format(self.num_classes)
        res_str += "\tTotal Samples: {}\n".format(self.x.shape[0])
        res_str += "\tVector size: {}\n".format(self.x.shape[1])
        res_str += "\tTrain percentage: {}%\n".format(int(self.ratio * 100))
        return res_str

    def plot_data(self, overlapped=False, showfig=True, save_path=None, library='matplotlib'):
        """
        Generates a figure with an example of the data
        :param overlapped: (boolean) If True it will plot all the examples in the same figure changing the color.
            Otherwise it will create a subplot with as many subplots as classes. Default: False
        :return: (fig, ax) matplolib format to be plotted.
        """
        if library == 'matplotlib':
            self._plot_data_matplotlib(overlapped=overlapped, showfig=showfig, save_path=save_path)
        elif library == 'plotly':
            self._plot_data_plotly(showfig=showfig, save_path=save_path)
        else:
            logger.warning("Unrecognized library to plot " + library)
            return None

    def _plot_data_plotly(self, showfig=False, save_path=None, extension=".svg"):
        fig = go.Figure()
        labels = self.y
        if self.categorical:
            labels = self.categorical_to_sparse(labels)
        for cls in range(self.num_classes):
            for index, label in enumerate(labels):
                if label == cls:
                    fig.add_trace(go.Scatter(x=np.real(self.x[index]), y=np.imag(self.x[index]),
                                             mode='markers', marker_symbol=cls, marker_size=10,
                                             name="Class: " + str(cls)))
                    break
        fig.update_layout(title='Data Visualization Example',
                          yaxis=dict(scaleanchor="x", scaleratio=1),
                          xaxis_title='real (x)',
                          yaxis_title='imaginary (y)',
                          showlegend=True)
        if save_path is not None:
            # https://plot.ly/python/configuration-options/
            os.makedirs(save_path, exist_ok=True)
            plotly.offline.plot(fig, filename=str(save_path / "data_example.html"), config={'editable': True},
                                auto_open=showfig)
            fig.write_image(str(save_path / "data_example") + extension)
        elif showfig:
            fig.show(config={'scrollZoom': True, 'editable': True})

    def _plot_data_matplotlib(self, overlapped=False, showfig=False, save_path=None, extension=".svg"):
        labels = self.y
        if self.categorical:
            labels = self.categorical_to_sparse(labels)
        if overlapped:
            fig, ax = plt.subplots()
            for cls in range(self.num_classes):
                for index, label in enumerate(labels):
                    if label == cls:
                        ax.plot(np.real(self.x[index]),
                                np.imag(self.x[index]),
                                MARKERS[cls % len(MARKERS)],
                                color=COLORS[cls % len(COLORS)],
                                label="Class " + str(cls)
                                )
                        ax.axis('equal')
                        ax.grid(True)
                        ax.set_aspect('equal', adjustable='box')
                        break
            ax.set_xlabel('real (x)')
            ax.set_ylabel('imaginary (y)')
            ax.legend(loc='upper right')
        else:
            fig, ax = plt.subplots(self.num_classes)
            for cls in range(self.num_classes):
                for index, label in enumerate(labels):
                    if label == cls:  # This is done in case the data is shuffled.
                        ax[cls].plot(np.real(self.x[index]),
                                     np.imag(self.x[index]), 'b.')
                        ax[cls].axis('equal')
                        ax[cls].grid(True)
                        ax[cls].set_aspect('equal', adjustable='box')
                        break
        if showfig:
            fig.show()
        if save_path is not None:
            save_path = Path(save_path)
            os.makedirs(save_path, exist_ok=True)
            prefix = ""
            if overlapped:
                prefix = "overlapped_"
            fig.savefig(save_path / Path(prefix + self.dataset_name + "_data_example" + extension), transparent=True)
            tikzplotlib.save(save_path / (prefix + self.dataset_name + "_data_example.tikz"))
        return fig, ax

    # =======
    # Getters
    # =======

    def get_train_and_test(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_test(self):
        return self.x_test, self.y_test

    def get_all(self):
        return self.x, self.y

    def get_categorical_labels(self):
        return self.sparse_into_categorical(self.y, self.num_classes)

    # ================
    # Static functions
    # ================

    @staticmethod
    def sparse_into_categorical(spar, num_classes=None):
        if len(spar.shape) == 1:  # Check data is indeed sparse
            spar = spar.astype(int)
            if num_classes is None:
                num_classes = max(spar) + 1  # assumes labels starts at 0
            cat = np.zeros((spar.shape[0], num_classes))
            for i, k in enumerate(spar):
                cat[i][k] = 1
        else:
            # Data was already categorical (I think)
            cat = spar
        return cat

    @staticmethod
    def categorical_to_sparse(cat):
        return np.argmax(cat, axis=1)

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

    @staticmethod
    def _get_next_batch(x, y, start, end):
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
        if start > end:  # What will happen if not? Should I leave this case anyway and just give a warning?
            sys.exit("Error:get_next_batch(): end should be higher than start")
        # TODO: Check end < len(x)
        x_batch = x[start:end]
        y_batch = y[start:end]
        return x_batch, y_batch


class OpenDataset(Dataset):
    """
    This class is used to init a Dataset with a saved npy data instead of giving the vector directly.
    Construction overload (either use vector x and y or use a string path)
        will be maybe cleaner but it does not exist in Python :S
    """

    def __init__(self, path, num_classes=None, ratio=0.8, savedata=False):
        self.path = cast_to_path(path)
        x, y = self.load_dataset(self.path)
        super().__init__(x, y, num_classes=num_classes, ratio=ratio, savedata=savedata,
                         dataset_name="opened dataset " + str(self.path))

    @staticmethod
    def load_dataset(path):
        try:
            x = np.load(path / "data.npy")
            y = np.load(path / "labels.npy")
        except FileNotFoundError:
            sys.exit("OpenDataset::load_dataset: Files data.npy and labels.npy not found in " + path)
        return x, y

    def summary(self, res_str=None):
        res_str = "Opened data located in {}\n".format(str(self.path))
        return super().summary(res_str)


class GeneratorDataset(ABC, Dataset):
    """
    Is a database method with an automatic x and y (data) generation.
    Used to automate the generation of data.
    Must therefore define a method to generate the data.

    Good Practice: Although it is not compulsory,
        it is recommended to define it's own summary method to know how the dataset was generated.
    """

    def __init__(self, m, n, num_classes=2, ratio=0.8, savedata=False, debug=False, dataset_name=None):
        """
        This class will first generate x and y with it's own defined method and then initialize a conventional dataset
        """
        x, y = self._generate_data(m, n, num_classes)
        x, y = randomize(x, y)
        if dataset_name is None:
            dataset_name = "Generated dataset"
        Dataset.__init__(self, x, y, num_classes=num_classes, ratio=ratio, savedata=savedata, debug=debug,
                         dataset_name=dataset_name)

    @abstractmethod
    def _generate_data(self, num_samples_per_class, num_samples, num_classes):
        """
        Abstract method. It MUST be defined.
        Method on how to generate x and y.
        """
        pass


class CorrelatedGaussianNormal(GeneratorDataset):

    def __init__(self, m, n, cov_matrix_list, num_classes=None, ratio=0.8, debug=False, savedata=False,
                 dataset_name=None, sort: bool = False):
        self.sort = sort
        if num_classes is None:
            num_classes = len(cov_matrix_list)
        if not len(cov_matrix_list) == num_classes:
            logger.error("cov_matrix_list length ({0}) should have the same size as num_classes ({1})".format(
                len(cov_matrix_list), num_classes))
            sys.exit(-1)
        for cov_mat in cov_matrix_list:  # Each class has a coviariance matrix 2x2
            # Numpy cast enables data to be either numpy array or list
            if not np.array(cov_mat).shape == (2, 2):
                logger.error("covariance matrix must have shape 2x2 but has shape {}".format(np.array(cov_mat).shape))
                sys.exit(-1)
            if not cov_mat[0][1] == cov_mat[1][0]:
                logger.error("Elements outside the diagonal must be equal (they are both sigma_{xy}")
                sys.exit(-1)
            if not np.abs(cov_mat[0][1] / sqrt(cov_mat[0][0] * cov_mat[1][1])) < 1:
                logger.error("corelation coefficient module must be lower than one")
                sys.exit(-1)
        self.cov_matrix_list = cov_matrix_list
        if dataset_name is None:
            dataset_name = "Correlated Gaussian Normal"
        super().__init__(m, n, num_classes=num_classes, ratio=ratio, savedata=savedata, debug=debug,
                         dataset_name=dataset_name)

    @staticmethod
    def _create_correlated_gaussian_point(num_samples, r=None, sort=False):
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
        y = [y[0][i] + 1j * y[1][i] for i in range(y.shape[1])]
        if sort:
            y.sort(key=lambda x: np.abs(x))
        # tmp = [np.abs(y[i]) < np.abs(y[i+1]) for i in range(0, len(y)-1)]
        # assert np.all(tmp)
        # set_trace()
        return y

    def _generate_data(self, num_samples_per_class, num_samples, num_classes):
        x = []
        y = []
        for signal_class in range(num_classes):
            r = self.cov_matrix_list[signal_class]
            y.extend(signal_class * np.ones(num_samples_per_class))
            for _ in range(num_samples_per_class):
                x.append(self._create_correlated_gaussian_point(num_samples, r, sort=self.sort))
        return np.array(x), np.array(y)

    def summary(self, res_str=None):
        res_str = "Correlated Gaussian Noise\n"
        for cls in range(self.num_classes):
            res_str += "class {}\n".format(cls)
            res_str += "\tPearson correlation coefficient: {}\n".format(self.get_coef_correl(cls))
            res_str += "\tsigma_x^2 = " + str(self.cov_matrix_list[cls][0][0]) + \
                       "sigma_y^2 = " + str(self.cov_matrix_list[cls][1][1])
            res_str += "\tCircularity quotient: {}\n".format(self.get_circularity_quotient(cls))
            variance, pseudo_variance = self.get_variance_and_pseudo_variance(cls)
            res_str += "\t\tvariance: {0}; pseudo-variance: {1}\n".format(variance, pseudo_variance)
            epsilon, alpha = self.get_ellipse_params(cls, deg=True)
            res_str += "\tEllipse epsilon: {0}, Angle (alpha): {1} deg\n".format(epsilon, alpha)
        return super().summary(res_str)

    # =====================================================
    # Get all the parameter equivalents
    # https://ieeexplore.ieee.org/abstract/document/4682548
    # =====================================================

    def get_coef_correl(self, index):
        rho = None
        if index < len(self.cov_matrix_list):
            cov_mat = self.cov_matrix_list[index]
            rho = cov_mat[0][1] / (sqrt(cov_mat[0][0]) * sqrt(cov_mat[1][1]))
        else:
            logger.error("Index out of range")
        return rho

    def get_variance_and_pseudo_variance(self, index):
        variance = None
        pseudo_variance = None
        if index < len(self.cov_matrix_list):  # TODO: Get the error here
            variance = self.cov_matrix_list[index][0][0] + self.cov_matrix_list[index][1][1]
            pseudo_variance = self.cov_matrix_list[index][0][0] - self.cov_matrix_list[index][1][1] + \
                              2j * self.cov_matrix_list[index][0][1]
        return variance, pseudo_variance

    def get_circularity_quotient(self, index):
        variance, pseudo_variance = self.get_variance_and_pseudo_variance(index)
        return pseudo_variance / variance

    def get_ellipse_params(self, index, deg=False):
        varrho = self.get_circularity_quotient(index)
        epsilon = sqrt(np.abs(varrho))
        alpha = np.angle(varrho, deg)
        return epsilon, alpha


class CorrelatedGaussianCoeffCorrel(CorrelatedGaussianNormal):

    def __init__(self, m, n, param_list, num_classes=None, ratio=0.8, debug=False, savedata=False, dataset_name=None,
                 sort: bool = False):
        if num_classes is None:
            num_classes = len(param_list)
        if not len(param_list) == num_classes:
            logger.error("param_list length ({0}) should have the same "
                         "size as num_classes ({1})".format(len(param_list), num_classes))
        cov_mat_list = []
        for param in param_list:
            if not len(param) == 3:
                logger.error("Each parameter in param_list should have size 3 "
                             "(coef correl and both variances) but {} where given".format(len(param)))
                sys.exit(-1)
            sigma_xy = param[0] * sqrt(param[1] * param[2])
            cov_mat_list.append([[param[1], sigma_xy], [sigma_xy, param[2]]])
        super().__init__(m=m, n=n, cov_matrix_list=cov_mat_list,
                         num_classes=num_classes, ratio=ratio, debug=debug, savedata=savedata,
                         dataset_name=dataset_name, sort=sort)


class ComplexNormalVariable(CorrelatedGaussianNormal):
    """
    This class is a correlated gaussian normal but instead of defining it's classes with the covariance matrix
    it is defined with it's complex variance and pseudo-variance.
    https://ieeexplore.ieee.org/abstract/document/4682548
    """

    def __init__(self, m, n, param_list, num_classes=None, ratio=0.8, debug=False, savedata=False):
        if num_classes is None:
            num_classes = len(param_list)
        if not len(param_list) == num_classes:
            logger.error("param_list length ({0}) should have the same size "
                         "as num_classes ({1})".format(len(param_list), num_classes))
            sys.exit(-1)
        cov_mat_list = []
        for param in param_list:
            if not len(param) == 2:
                logger.error("Each parameter in param_list should have size 2 "
                             "(sigma and tau) but {} where given".format(len(param)))
                sys.exit(-1)
            cov_mat_list.append(self.get_cov_matrix(param[0], param[1]))
        super().__init__(m=m, n=n, cov_matrix_list=cov_mat_list,
                         num_classes=num_classes, ratio=ratio, debug=debug, savedata=savedata)
        for i, param in enumerate(param_list):  # Just for fun
            assert self.get_circularity_quiotient(i) == param[1] / param[0], \
                "ComplexNormalVariable::__init__: Error in creating data"

    @staticmethod
    def get_cov_matrix(sigma, tau):
        sigma_xx = (sigma + np.real(tau)) / 2
        sigma_yy = (sigma - np.real(tau)) / 2
        sigma_xy = np.imag(tau) / 2
        return [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]]


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
        super().__init__(m, n, num_classes=num_classes, ratio=ratio, savedata=savedata, dataset_name="Gaussian Noise")

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
            logger.info("Class " + str(k) + ": mu = " + str(mu) + "; sigma = " + str(sigma))
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


# =================
# Testing Functions
# =================


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
        dataset = CorrelatedGaussianNormal(m, n, num_classes=num_classes, debug=False, cov_matrix_list=coef)
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


def sup(a, b):
    return a > b


def inf(a, b):
    return a > b


def get_parametric_predictor_labels(x, y, coef_1=0.5, coef_2=-0.5):
    rho = []
    for re, im in zip(x, y):
        cov = np.cov([re, im])
        rho.append(cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1])))

    rho = np.array(rho)
    thresh = np.full(rho.shape, (coef_1 + coef_2) / 2)
    if coef_1 > coef_2:
        result = np.less(rho, thresh).astype(int)
    else:
        result = np.greater(rho, thresh).astype(int)
    return result


def parametric_predictor(dataset, coef_1=0.5, coef_2=-0.5):
    x = np.real(dataset.x)
    y = np.imag(dataset.x)
    result = get_parametric_predictor_labels(x=x, y=y, coef_1=coef_1, coef_2=coef_2)
    acc = np.sum(np.equal(Dataset.categorical_to_sparse(dataset.y), result)) / len(result)
    return acc


if __name__ == "__main__":
    # create_subplots_of_graph()
    m = 10000
    n = 128
    """cov_matr_list = [
        [[1, 0.75], [0.75, 1]],
        [[1, -0.75], [-0.75, 1]]
    ]
    dataset = CorrelatedGaussianNormal(m, n, cov_matr_list, debug=False)"""
    for coef in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        dataset = CorrelatedGaussianCoeffCorrel(m, n, param_list=[[coef, 1, 1], [-coef, 1, 1]],
                                                dataset_name=f"{int(coef*100)}")
        dataset.plot_data(overlapped=True, showfig=False,
                          save_path="/home/barrachina/Dropbox/thesis/CVNN-thesis-Agustin/ppts/20210611 - ICASSP/img")
    # dataset.save_data("./data/MLSP/")
    # print(parametric_predictor(dataset))

    # dataset = OpenDataset("./data/MLSP/")
    # dataset.plot_data(overlapped=True, showfig=True, library="matplotlib")
    # set_trace()
    # print("{:.2%}".format(parametric_predictor(dataset)))

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.1.23'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
