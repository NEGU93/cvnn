.. _real_vs_complex:

Real Vs Complex
---------------

.. py:class:: RealVsComplex(MonteCarlo)

    Inherits from MonteCarlo. It generates the same files with the exception that the excel file is called :code:`./log/rvnn_vs_cvnn_monte_carlo_summary.xlsx`

    Compares a complex model with it's real equivalent.

    Example usage::

        # Assume you already have complex data 'x' with its labels 'y'... and a Cvnn model.

        montecarlo = RealVsComplex(complex_model)
        montecarlo.run(x, y)


.. py:method:: __init__(self, complex_model, capacity_equivalent=True, equiv_technique='ratio')

    Used to compare a single Complex Model given as a parameter. The Code will generate it's real equivalent and compre both of them.

    :param complex_model: :code:`tensorflow.keras.model`
    :param capacity_equivalent: An equivalent model can be equivalent in terms of layer neurons or trainable parameters (capacity equivalent according to `this paper <https://arxiv.org/abs/1811.12351>`_)
        
        - True, it creates a capacity-equivalent model in terms of trainable parameters
        - False, it will double all layer size (except the last one if classifier=True)
    :param equiv_technique: Used to define the strategy of the capacity equivalent model.
        This parameter is ignored if :code:`capacity_equivalent=False`
        
        - 'ratio': :code:`neurons_real_valued_layer[i] = r * neurons_complex_valued_layer[i]`, 'r' constant for all 'i'
        - 'alternate': Method described in `this paper <https://arxiv.org/abs/1811.12351>`_ where one alternates between multiplying by 2 or 1. Special case on the middle is treated as a compromise between the two.
