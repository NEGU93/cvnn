import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import re
import os
from pdb import set_trace

"""
Monte Carlo csv files
---------------------
saved on ./results/histogram_iter[0-9]+_classes[0-9]+.csv
"""


def plot_csv_histogram(path, filename, column=None, visualize=False):
    """
    Opens a csv file and creates a png file with the histogram of the csv result.
    This is used to make many simulations of both RVNN and CVNN and compare them with statistics.
    :param path: Path where the csv file is located.
    :param filename: Name of the csv file
    :param column:
    :param visualize: True to show the image saved
    :return: None
    """
    # https://medium.com/python-pandemonium/data-visualization-in-python-histogram-in-matplotlib-dce38f49f89c
    assert type(filename) == str
    data = pd.read_csv(path + filename)

    data.hist(column=column, bins=data.shape[0]//10)
    if column is not None:
        filename = column + filename
    plt.savefig(path + filename.replace('.csv', '.png'))    # Save the image with same name as csv
    if visualize:
        plt.show()

    mean = data.mean()
    std = data.std()
    set_trace()


def get_trailing_number(s):
    m = re.search(r'\d+.csv$', s)   # I get only the end of the string (last number of any size and .csv extension)
    # splitext gets root [0] and extension [1] of the name.
    return int(os.path.splitext(m.group())[0]) if m else None


def get_histogram_results(path):
    project_path = os.path.abspath(path)
    list_of_files = glob.glob(project_path + "/*[0-9].csv")     # csv files that will end with a number.
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
            # set_trace()
            cvnn_loss += dic['CVNN loss']
            rvnn_loss += dic[' RVNN loss']       # TODO: this will be a problem in the future! (space before string)
            cvnn_acc += dic[' CVNN acc']
            rvnn_acc += dic[' RVNN acc']
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
__version__ = '1.0.9'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
