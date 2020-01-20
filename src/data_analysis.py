import matplotlib.pyplot as plt
import pandas as pd


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


__author__ = 'J. Agustin BARRACHINA'
__version__ = '1.0.0'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
