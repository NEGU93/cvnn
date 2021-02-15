Monte Carlo class has an dictionary of booleans values that let's you configure the output files on a monte carlo run::

    output_config = {
            'plot_all': False,
            'confusion_matrix': False,
            'excel_summary': True,
            'debug': False,
            'summary_of_run': True,
            'safety_checkpoints': False
        }

.. csv-table:: Output files 
   :file: output_table.csv
   :widths: 20, 20, 60
   :header-rows: 1