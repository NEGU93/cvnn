import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_csv_histogram(path, filename, column=None, visualize=False):
    # https://medium.com/python-pandemonium/data-visualization-in-python-histogram-in-matplotlib-dce38f49f89c
    assert type(filename) == str
    data = pd.read_csv(path + filename)

    data.hist(column=column, bins=data.shape[0]//10)
    if column is not None:
        filename = column + filename
    plt.savefig(path + filename.replace('.csv', '.png'))
    if visualize:
        plt.show()


def sparse_confusion_matrix(y_pred_np, y_label_np):
    y_pred_pd = pd.Series(y_pred_np, name='Predicted')
    y_label_pd = pd.Series(y_label_np, name='Actual')

    return pd.crosstab(y_label_pd, y_pred_pd, rownames=['Actual'], colnames=['Predicted'], margins=True)


def categorical_confusion_matrix(y_pred_np, y_label_np):
    return sparse_confusion_matrix(np.argmax(y_pred_np, axis=1), np.argmax(y_label_np, axis=1))


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


__author__ = 'J. Agustin BARRACHINA'
__version__ = '1.0.2'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
