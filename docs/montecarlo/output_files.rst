.. _output-files:

Output files
------------

Monte Carlo class will generate at least four different type of files on the path :file:`./logs/montecarlo/<year>/<month>/<day>/run_<time>/`

1. :file:`run_data.csv`: Raw results with the :code:`loss`, :code:`epoch`, :code:`model`, etc.

2. :file:`<model_name>_statistical_result.csv`: One file is generated for each model in the monte carlo simulation. It contains the statistical results per epoch of the model.

3. :file:`models_details.json`: A full detailed description of each model to be trained. 

4. :file:`run/iteration<iteration>_model<model index and name>/<model_name>_results_fit.csv`: Inside the :file:`run` folder there is information of the result for each model at each iteration.

There are many other optional files that can be controled using the `output_config` dictionary variable of booleans::

    output_config = {
            'plot_all': False,
            'confusion_matrix': False,
            'excel_summary': True,
            'summary_of_run': True,
            'tensorboard': False,
            'save_weights': False,
            'safety_checkpoints': False
        }

A complementary file :file:`test_results.csv` can be generated if :code:`test_data` is passed :meth:`run()` (:code:`None` by default). 

Usage example::

    monte_carlo = MonteCarlo()
    monte_carlo.output_config['plot_all'] = True      # Tell monte carlo to save the plots


.. csv-table:: Output files 
   :file: output_table.csv
   :widths: 20, 20, 60
   :header-rows: 1
