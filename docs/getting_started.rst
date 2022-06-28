Getting Started
===============

Welcome to my library! [CIT2019-BARRACHINA-CODE]_

**Ideology**

The idea of this library is just to implement :doc:`Complex layers <layers>` (:code:`ComplexLayer`) so that everything else stays the same as any Tensorflow code.

The only difference with a Tensorflow code is that you should use :code:`cvnn.layers` module instead of :code:`tf.keras.layers`.

Although :code:`tf.activation` and :code:`tf.initializers` could be used, it is HIGHLY recommended (and some times compulsory) to use the :code:`cvnn` module options.

.. warning::
        For a reason I ignore, TensorFlow casts the input automatically to floating. To avoid this, always create first a :code:`ComplexInput` layer in all your models.

If you are here is because you want to train a Complex-Valued Neural Network (CVNN). Use the following link for a quick tutorial.
    
    - :doc:`1.1. You want to train a Complex Valued Model as you will do with Keras <cvnn>`
        

**Real-valued case**

Although this library is intended to work with complex data type, it also supports real dtype by just using the :code:`dtype` parameter in each :code:`ComplexLayer` constructor.

All :code:`cvnn` activation functions and initializers already work Ok for real and complex inputs, so nothing should change.

This allows me to debug the code (comparing it's result with keras on real data) but also to easily implement a comparison between a complex and a real network minimizing the error.

You have some examples of this as for example :doc:`MNIST <code_examples/mnist_example>`.

.. note:: 
    Please, remember to `cite me <https://github.com/NEGU93/cvnn#cite-me>`_ accordingly [CIT2019-BARRACHINA-CODE]_

.. [CIT2019-BARRACHINA-CODE] Jose Agustin Barrachina. "Complex-Valued Neural Networks (CVNN)". GitHub repository. 
