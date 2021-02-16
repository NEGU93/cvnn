.. _plotter:

Plotter
-------

.. py:class:: Plotter

    This class manages the plot of results for a model train.
    It opens the csv files (test and train) saved during training and plots results as wrt each epoch saved.
    This class is generally used to plot accuracy and loss evolution during training.

.. py:method:: __init__(self, path, file_suffix: str = "_results_fit.csv", data_results_dict: dict = None, model_name: str = None)

    :path: Full path where the csv results are stored
    :file_suffix: (Optional) let's you filter csv files to open only files that ends with the suffix. By default it opens every csv file it finds.

.. py:method:: plot_key(self, key='loss', reload=False, library='plotly', showfig=False, savefig=True, index_loc=None, extension=".svg")

    :param library: String stating the library to be used to generate the box plot.

        - `matplotlib <https://matplotlib.org/stable/index.html>`_
        - `plotly <https://plotly.com/python/>`_
    :param key: String stating what to plot using tf.keras.History labels. ex. `val_accuracy` for the validation acc
    :param showfig: If True, it will show the grated box plot
    :param savefig: If True, it saves the figure at `self.path/key_library.extension`
    :param reload: If True it will reload data from the csv file in case it has changed.
    :param index_loc:
    :param extension: file extensions (default svg) to be used when saving the file (ignored if library is plotly).