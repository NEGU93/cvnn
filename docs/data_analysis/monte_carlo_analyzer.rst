.. _monte-carlo-analyzer:

Monte Carlo Analyzer
--------------------

.. note:: All seaborn and matplotlib saved figures also generates a tikz figure (.tex) for LaTeX reports.

**Small code example**

.. code-block:: python

    path = "path/to/the/run_data/file"
    monte = MonteCarloAnalyzer(path=path)
    monte.do_all(showfig=False, savefig=True)   # Generates some plots

.. py:class:: MonteCarloAnalyzer

    This class helps organizing Monte Carlo class results and plotting them.
    his class works with the run_data.csv generated with :code:`cvnn.montecarlo.MonteCarlo` class :ref:`montecarlo_class`

.. py:method:: __init__(self, df=None, path=None, history_dictionary: Optional[dict] = None)

    There are 2 ways to use this class:

    1. Either give the data as a `pandas <https://pandas.pydata.org/>`_ :code:`DataFrame`.
    2. Or give a file path to the :code:`run_data.csv` file generated with :code:`cvnn.montecarlo.MonteCarlo` class.
    
    The class will generate the corresponding csv file (if option 1) or obtain the dataframe from the csv file (option 2)
    
    :param df: (Optional) `pandas <https://pandas.pydata.org/>`_ :code:`DataFrame` with the data to be plotted.
    :param path: (Optional)

       1. If df was given, this can be the a path for MonteCarloAnalyzer to save a :code:`run_data.csv` file. If path is not given, it will use the default path :code:`./log/montecarlo/<year>/<month>/<day>/run_<time>/`
       1. If df is not given, path should be:
           - The full path and filename for the run_data.csv to be plotted
           - A path to search of ALL :code:`run_data.csv` that it can find (even within subfolders). This is useful when you want to plot together different :code:`MonteCarlo.run()` results. For example, it enables to run two simulations of 50 iterations each and plot them as if it was a single run of 100 iterations.
    :param history_dictionary: (Optional) dictionary. This parameter is only used if df and path are None. Dictionary with the models names as keys and a list of full paths to the model history pickle file.

.. py:method:: do_all(self, extension=".svg", showfig=False, savefig=True)

    Plots :meth:`box_plot`, :meth:`plot_histogram` and confidence interval (using :code:`MonteCarloPlotter`) for both `plotly <https://plotly.com/python/>`_ and `seaborn <https://seaborn.pydata.org/>`_ libraries for keys :code:`val_accuracy`, :code:`val_loss`, :code:`accuracy` and :code:`loss`.

.. py:method:: box_plot(self, epoch=-1, library='plotly', key='val_accuracy', showfig=False, savefig=True, extension='.svg')

    Saves/shows a box plot of the results. [BOX-PLOT]_

    :param epoch: Which epoch to use for the box plot. If :code:`-1` (default) it will use the last epoch.
    :param library: string stating the library to be used to generate the box plot. Either `plotly <https://plotly.com/python/>`_ or `seaborn <https://seaborn.pydata.org/>`_
    :param key: String stating what to plot using :code:`tf.keras.History` labels. ex. :code:`val_accuracy`, :code:`val_loss`, :code:`accuracy` or :code:`loss`.
    :param showfig: If True, it will show the grated box plot
    :param savefig: If True, it saves the figure at: :code:`self.path / "plots/box_plot/"`
    :param extension: file extensions (default svg) to be used when saving the file (only used when library is seaborn).

**Output example using pyplot**

.. raw:: html

   <iframe src="../_static/data_analysis_examples/montecarlo_test_accuracy_box_plot.html" height="500px" width="100%"></iframe>

**Output example using seaborn**

.. raw:: html

    <object data="../_static/data_analysis_examples/montecarlo_test_accuracy_box_plot.svg" type="image/svg+xml"></object>

.. py:method:: plot_histogram(self, key='val_accuracy', epoch=-1, library='seaborn', showfig=False, savefig=True, title='', extension=".svg")

    Saves/shows a histogram of the results.

    :param epoch: Which epoch to use for the histogram. If :code:`-1` (default) it will use the last epoch.
    :param library: string stating the library to be used to generate the box plot:
    
        - `matplotlib <https://matplotlib.org/stable/index.html>`_ 
        - `plotly <https://plotly.com/python/>`_
        - `seaborn <https://seaborn.pydata.org/>`_
    :param key: String stating what to plot using :code:`tf.keras.History` labels. ex. :code:`val_accuracy`, :code:`val_loss`, :code:`accuracy` or :code:`loss`.
    :param showfig: If True, it will show the grated box plot
    :param savefig: If True, it saves the figure at: :code:`self.path / "plots/box_plot/"`
    :param title: Figure title
    :param extension: file extensions (default svg) to be used when saving the file (ignored if library is plotly).

**Output example using pyplot**

.. raw:: html

   <iframe src="../_static/data_analysis_examples/montecarlo_test_accuracy_histogram.html" height="500px" width="100%"></iframe>

**Output example using seaborn**

.. raw:: html

    <object data="../_static/data_analysis_examples/histogram_montecarlo_test_accuracy_seaborn.svg" type="image/svg+xml"></object>

**Output example using matplotlib**

.. raw:: html

    <object data="../_static/data_analysis_examples/montecarlo_te_histogram.svg" type="image/svg+xml"></object>

.. [BOX-PLOT] Williamson, David F., Robert A. Parker, and Juliette S. Kendrick. "The box plot: a simple visual method to interpret data." Annals of internal medicine 110.11 (1989): 916-921.