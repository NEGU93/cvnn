import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pdb import set_trace


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


"""-------------
Confusion Matrix
-------------"""


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


def plot_confusion_matrix(y_pred_np, y_label_np, title='Confusion matrix', cmap=plt.cm.gray_r):
    df_confusion = sparse_confusion_matrix(y_pred_np, y_label_np)
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


if __name__ == '__main__':
    plot_loss_and_acc("/home/barrachina/Documents/cvnn/log/CVNN_testing/run-20200127140842/CVNN_testing.csv"
                      , visualize=True)

__author__ = 'J. Agustin BARRACHINA'
__version__ = '1.0.5'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
