.. _plotter:

Plotter
-------

.. py:class:: Plotter

    This class manages the plot of results for a model train.
    It opens the csv files (test and train) saved during training and plots results as wrt each epoch saved.
    This class is generally used to plot accuracy and loss evolution during training.

.. py:method:: __init__(self, path, file_suffix: str = "_results_fit.csv", data_results_dict: dict = None, model_name: str = None)

    :path: Full path where the csv results are stored
    :file_suffix: (optional) let's you filter csv files to open only files that ends with the suffix.
        By default it opens every csv file it finds.