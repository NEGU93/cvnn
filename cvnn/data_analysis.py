import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import re
import os
from pdb import set_trace
import scipy.stats as stats


def find_intersection_of_gaussians(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a, b, c])


def plot_gaussian(mu=0, std=1, x=None,
                  y_label=None, x_label=None, title=None,
                  filename="./results/plot_2_gaussian_output.png", showfig=False, savefig=True):
    """
    Plot a gaussian function
    :param mu: mean of the gaussian
    :param std:  Standard deviation of the gaussian
    :param x: The x axis linespace data. If None, an automatic one will be generated. Default is None
    :param y_label: The y axis label.
    :param x_label: The x axis label.
    :param title: str or None. The legend’s title. Default is no title (None).
    :param filename: Only used when savefig=True. The name of the figure to be saved
    :param showfig: Boolean. If true it will show the figure using matplotlib show method
    :param savefig: Boolean. If true it will save the figure with the name of filename parameter
    :return: tuple (fig, ax) from the plotted figure
    """
    if x is None:
        x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    fig, ax = plt.subplots()
    ax.plot(x, stats.norm.pdf(x, mu, std))

    # Figure parameters
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        fig.suptitle(title)

    # save/show results
    if savefig:
        fig.savefig(filename)
    if showfig:
        fig.show()
    return fig, ax


def plot_2_gaussian(mu_1, std_1, mu_2, std_2, name_1, name_2, x=None,
                    y_label=None, x_label=None, loc=None, title=None,
                    filename="./results/plot_2_gaussian_output.png", showfig=False, savefig=True):
    """
    Plots 2 gaussians on the same plot using matplotlib
    :param mu_1: Mean of first gaussian to be plotted
    :param std_1: Standard deviation of the first gaussian to be plotted
    :param mu_2: Mean of second gaussian to be plotted
    :param std_2: Standard deviation of the second gaussian to be plotted
    :param name_1: Name of the first gaussian (Used in legend)
    :param name_2: Name of the second gaussian (Used in legend)
    :param x: The x axis linespace data. If None, an automatic one will be generated. Default is None
    :param y_label: The y axis label.
    :param x_label: The x axis label.
    :param loc: can be a string or an integer specifying the legend location. default: None.
                    https://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend
    :param title: str or None. The legend’s title. Default is no title (None).
    :param filename: Only used when savefig=True. The name of the figure to be saved
    :param showfig: Boolean. If true it will show the figure using matplotlib show method
    :param savefig: Boolean. If true it will save the figure with the name of filename parameter
    :return: tuple (fig, ax) from the plotted figure
    """
    # Get x axes
    ax_min = min(mu_1 - 3 * std_1, mu_2 - 3 * std_2)
    ax_max = max(mu_1 + 3 * std_1, mu_2 + 3 * std_2)

    # Get gaussian data
    if x is None:
        x, dx = np.linspace(ax_min, ax_max, 100, retstep=True)
    gauss_1 = stats.norm.pdf(x, mu_1, std_1)
    gauss_2 = stats.norm.pdf(x, mu_2, std_2)

    # plot gaussians
    fig, ax = plt.subplots()
    ax.plot(x, gauss_1, label=name_1)
    ax.plot(x, gauss_2, label=name_2)

    # plot intersection point
    result = find_intersection_of_gaussians(mu_1, mu_2, std_1, std_2)
    keep_root = [r for r in result if ax_max > r > ax_min]
    for k in keep_root:
        intersection_point = stats.norm.pdf(k, mu_1, std_1)
        ax.plot(k, intersection_point, 'o')
        ax.hlines(intersection_point, xmin=ax_min, xmax=k, linestyle='--', color='black')
        ax.vlines(k, ymin=0, ymax=intersection_point, linestyles='--', color='black')
        trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, intersection_point, "{:.2f}".format(intersection_point),
                transform=trans, ha="right", va="center")

    # Figure parameters
    if loc is not None:
        fig.legend(loc=loc)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set(xlim=(ax_min, ax_max), ylim=(0, max(max(gauss_1), max(gauss_2)) * 1.05))
    if title is not None:
        fig.suptitle(title)

    # save/show results
    if savefig:
        fig.savefig(filename)
    if showfig:
        fig.show()
    return fig, ax


def plot_2_hist(data_1, data_2, name_1, name_2, bins=None, y_label='', x_label='', loc='upper right', title='',
                filename="./results/plot_2_hist_output.png", showfig=False, savefig=True):
    """
    Plot 2 histograms in the same figure using matplotlib
    :param data_1: Data for the first histogram
    :param data_2: Data for the second histogram
    :param name_1: Name of the first histogram (Used in legend)
    :param name_2: Name of the second histogram (Used in legend)
    :param bins:  int or sequence or str, optional
    :param y_label: The y axis label.
    :param x_label: The x axis label.
    :param loc: can be a string or an integer specifying the legend location. default: None.
                    https://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend
    :param title: str or None. The legend’s title. Default is no title (None).
    :param filename: Only used when savefig=True. The name of the figure to be saved
    :param showfig: Boolean. If true it will show the figure using matplotlib show method
    :param savefig: Boolean. If true it will save the figure with the name of filename parameter
    :return tuple (fig, ax) from the plotted figure
    """
    fig, ax = plt.subplots()
    if bins is None:
        bins = np.linspace(0, 1, 101)
    ax.hist(data_1, bins, alpha=0.5, label=name_1)
    ax.hist(data_2, bins, alpha=0.5, label=name_2)
    # Figure parameters
    if loc is not None:
        fig.legend(loc=loc)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        fig.suptitle(title)
    ax.axis(xmin=min(min(data_1), min(data_2)) - 0.01, xmax=max(max(data_1), max(data_2)) + 0.01)

    # save/show results
    if showfig:
        fig.show()
    if savefig:
        fig.savefig(filename)
    return fig, ax


"""
Monte Carlo csv files
---------------------
saved on ./results/histogram_iter[0-9]+_classes[0-9]+.csv
"""


def plot_csv_histogram_matplotlib(filename, bins=None, column=None, showfig=False):
    """
    Plots and saves a histogram image using the data from a csv file with help of matplotlib.
    :param filename: Full path + name of the csv file to be opened (must contain the csv extension)
                TODO: automatically add the extension if it's not there
    :param bins: int or sequence or str, optional
    :param column:
    :param showfig: Boolean. If true it will show the figure using matplotlib show method
    :return: None
    """
    assert type(filename) == str
    path, file = os.path.split(filename)
    data = pd.read_csv(path + filename)
    fig, ax = plot_2_hist(data['CVNN acc'], data['RVNN acc'], 'CVNN', 'RVNN', bins=bins, showfig=showfig, savefig=False)
    fig.savefig(path + "matplot_histogram_" + file.replace('.csv', '.png'))  # Save the image with same name as csv


def plot_csv_histogram_pandas(filename, bins=None, column=None, showfig=False):
    """
    Opens a csv file and creates a png file with the histogram of the csv result.
    This is used to make many simulations of both RVNN and CVNN and compare them with statistics.
    :param filename: Full path + name of the csv file to be opened (must contain the csv extension)
                TODO: automatically add the extension if it's not there
    :param bins: int or sequence or str, optional
    :param column:
    :param showfig: Boolean. If true it will show the figure
    :return: None
    """
    # https://medium.com/python-pandemonium/data-visualization-in-python-histogram-in-matplotlib-dce38f49f89c
    assert type(filename) == str
    path, file = os.path.split(filename)
    data = pd.read_csv(path + filename)
    if bins is None:
        bins = data.shape[0] // 10
    data.hist(column=column, bins=bins)
    if column is not None:
        filename = column + filename
    plt.savefig(path + "histogram_" + filename.replace('.csv', '.png'))  # Save the image with same name as csv
    if showfig:
        plt.show()


def get_trailing_number(s):
    """
    Search for a termination of a file name that has the ".csv" extension and that has a number at the end.
    It gives the number at the end of the file. This number can have any amount of digits.
    Example:
    x = get_trailing_number("my/path/to/file/any_43_start_name9872.csv")    # x = 9872
    y = get_trailing_number("my/path/to/file/any_43_start_name.csv")        # y = None
    z = get_trailing_number("my/path/to/file/any_43_start_name85498.txt")   # y = None
    :param s: The string to search for the specific term
    :return: The number located before the extension. None if there is no number.
    """
    m = re.search(r'\d+.csv$', s)  # I get only the end of the string (last number of any size and .csv extension)
    # splitext gets root [0] and extension [1] of the name.
    return int(os.path.splitext(m.group())[0]) if m else None


def get_histogram_results(path="./"):
    """

    :param path:
    :return:
    """
    project_path = os.path.abspath(path)
    list_of_files = glob.glob(project_path + "/*[0-9].csv")  # csv files that will end with a number.
    d = dict()
    if list_of_files is not None:
        for file in list_of_files:
            k = get_trailing_number(file)
            val = get_loss_and_acc_means(file)
            if val is not None:
                if k in d:
                    d[k].append(val)
                else:
                    d[k] = [val]
    return d


def get_pandas_mean_for_each_class(d):
    result = dict()
    for k in d.keys():
        cvnn_loss = 0
        rvnn_loss = 0
        cvnn_acc = 0
        rvnn_acc = 0
        for dic in d[k]:
            cvnn_loss += dic['CVNN loss']
            rvnn_loss += dic['RVNN loss']
            cvnn_acc += dic['CVNN acc']
            rvnn_acc += dic['RVNN acc']
        result[k] = {'CVNN loss': cvnn_loss / len(d[k]),
                     'RVNN loss': rvnn_loss / len(d[k]),
                     'CVNN acc': cvnn_acc / len(d[k]),
                     'RVNN acc': rvnn_acc / len(d[k])}
    return pd.DataFrame(result).transpose()


def get_loss_and_acc_means(filename):
    try:
        data = pd.read_csv(filename)
        return data.mean().to_dict()
    except pd.errors.EmptyDataError:
        print("pandas.errors.EmptyDataError: get_loss_and_acc_means: No columns to parse from file")
        return None


def get_loss_and_acc_std(filename):
    try:
        data = pd.read_csv(filename)
        return data.std().to_dict()     # TODO: can return Nan with only one data
    except pd.errors.EmptyDataError:
        print("pandas.errors.EmptyDataError: get_loss_and_acc_means: No columns to parse from file")
        return None


"""
Loss and Acc saved csv
----------------------
saved on ./log/<name>/run-<date>/<name>.csv
"""


def plot_loss_and_acc(filename, visualize=False):
    assert type(filename) == str
    data = pd.read_csv(filename)
    plt.plot(data['train loss'], 'o-', label='train loss')
    plt.plot(data['test loss'], label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(filename.replace('.csv', '_loss.png'))
    if visualize:
        plt.show()

    plt.figure()
    plt.plot(data['train acc'], 'o-', label='train acc')
    plt.plot(data['test acc'], label='test acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(filename.replace('.csv', '_acc.png'))
    if visualize:
        plt.show()

    set_trace()


"""
Confusion Matrix
----------------
Given results and labels directly
"""


def sparse_confusion_matrix(y_pred_np, y_label_np, filename=None):
    y_pred_pd = pd.Series(y_pred_np, name='Predicted')
    y_label_pd = pd.Series(y_label_np, name='Actual')
    df = pd.crosstab(y_label_pd, y_pred_pd, rownames=['Actual'], colnames=['Predicted'], margins=True)
    if filename is not None:
        fig, ax = plt.subplots()
        sns.heatmap(df,
                    annot=True,
                    linewidths=.5,
                    cbar=True,
                    )
        fig.savefig(filename)
    return df


def categorical_confusion_matrix(y_pred_np, y_label_np, filename=None):
    return sparse_confusion_matrix(np.argmax(y_pred_np, axis=1), np.argmax(y_label_np, axis=1), filename)


if __name__ == '__main__':
    res = get_histogram_results('./results')
    res = get_pandas_mean_for_each_class(res)
    # plot_loss_and_acc("/home/barrachina/Documents/cvnn/log/CVNN_testing/run-20200127140842/CVNN_testing.csv"
    # , visualize=True)
    set_trace()

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.12'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
