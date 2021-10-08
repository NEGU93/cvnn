Several Monte Carlo Comparison
------------------------------

**Small code example**

.. code-block:: python

    paths = [
        "path/to/relu/run/run_data",        # montecarlo result using relu
        "path/to/tanh/run/run_data",        # montecarlo result using tanh
        "path/to/sigmoid/run/run_data"      # montecarlo result using sigmoid
    ]
    several = SeveralMonteCarloComparison('Activation Functions', x=['ReLU', 'tanh', 'sigmoid'], paths=paths)
    several.box_plot(library='seaborn', showfig=True, savefile='path/to/save/result/act_fun_comparison_box_plot')

You can automate this code knowing that the return of each monte carlo run is the path.

.. code-block:: python

    paths = []
    # Run several Monte Carlo's
    for monte_carlo in monte_carlo_with_different_learning_rates:
        paths.append(monte_carlo.run(x, y))
    # Run self
    several = SeveralMonteCarloComparison('learning rate', x = learning_rates, paths =paths)
    several.box_plot(showfig=True)


.. py:class:: SeveralMonteCarloComparison

    This class is used to compare several monte carlo runs done with :class:`cvnn.montecarlo.MonteCarlo` class.
    It let's you compare different models between them but let's you not change other values like epochs.
    You can run as several MonteCarlo runs and then use :class:`SeveralMonteCarloComparison` class to compare the results.

.. py:method:: __init__(self, label, x, paths, round=2)

    :param label: string that describes what changed between each montecarlo run
    :param x: List of the value for each monte carlo run wrt :label:.
    :param paths: Full path to each monte carlo run_data saved file (Must end with run_data)

.. note::

    x and paths must be the same size


.. py:method:: box_plot(self, key='accuracy', library='plotly', epoch=-1, showfig=False, savefile=None)

    Saves/shows a box plot of the results.
        
    :param key: String stating what to plot using :class:`tf.keras.History` labels. ex. :code:`val_accuracy` for the validation accuracy.
    :param library: string stating the library to be used to generate the box plot.

        - `plotly <https://plotly.com/python/>`_
        - `seaborn <https://seaborn.pydata.org/>`_
    :param epoch: Which epoch to use for the box plot. If :code:`-1` (default) it will use the last epoch.
    :param showfig: If True, it will show the grated box plot.
    :param savefile: String with the path + filename where to save the boxplot. If :code:`None` (default) no figure is saved.

**Output example using pyplot**

.. raw:: html

   <iframe src="../_static/data_analysis_examples/several_box_plot.html" height="500px" width="100%"></iframe>

**Output example using seaborn**

.. raw:: html

    <object data="../_static/data_analysis_examples/dataset_dropout.svg" type="image/svg+xml"></object>