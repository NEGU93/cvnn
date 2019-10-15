import matplotlib.pyplot as plt
import pandas as pd


def plot_csv_histogram(filename):
    # https://medium.com/python-pandemonium/data-visualization-in-python-histogram-in-matplotlib-dce38f49f89c
    data = pd.read_csv(filename)
    data.hist(bins=20)
    plt.savefig(filename.replace('.csv', '.png'))
