Getting Started
===============

Welcome to my library! [CIT2019-BARRACHINA-CODE]_

**Ideology**

The idea of this library is just to implement :doc:`Complex layers <layers>` (:code:`ComplexLayer`) so that everything else stays the same as any Tensorflow code.

The only difference with a Tensorflow code is that you should use :code:`cvnn.ComplexLayers` module instead of :code:`tf.keras.layers`.

Although :code:`tf.activation` and :code:`tf.initializers` could be used, it is HIGHLY recommended (and some times compulsory) to use the `cvnn` module options.

.. warning::
        For a reason I ignore, TensorFlow casts the input automatically to floating. To avoid this, always create first a :code:`ComplexInput` layer in all your models.

If you are here is because you want to train a Complex-Valued Neural Network (CVNN). 
However, your situation may vary, this will help you find your use case and guide you how to use the library.
The recommended way to go would be 1.2. as the other may easily be done from that.

    1. You already have a complex dataset you want to use
    
        :doc:`1.1. You want to train a Complex Valued Model as you will do with Keras <cvnn>`
        
        1.2. You want to compare different models and do statistics with it's results
            - You want to feed each model to compare :ref:`montecarlo_class`.
            - You want a RVNN equivalent to be created automatically and compare it with the feeded CVNN :ref:`real_vs_complex`.
         
    :doc:`2. You don't have a specific dataset, you just want to play with this library <experiments/correlation_noise>`

**Real-valued case**

Although this library is intended to work with complex data type, it also supports real dtype by just using the :code:`dtype` parameter in each :code:`ComplexLayer` constructor.

All :code:`cvnn` activation functions and initializers already work Ok for real and complex inputs, so nothing should change.

This allows me to debug the code (comparing it's result with keras on real data) but also to easily implement a comparison between a complex and a real network minimizing the error.

.. note:: 
    Please, remember to `cite me <https://github.com/NEGU93/cvnn#cite-me>`_ accordingly [CIT2019-BARRACHINA-CODE]_

.. [CIT2019-BARRACHINA-CODE] Jose Agustin Barrachina. "Complex-Valued Neural Networks (CVNN)". GitHub repository. 
