Base Case Type A 2HL
^^^^^^^^^^^^^^^^^^^^

**Monte Carlo run**

- Iterations: 1000
- epochs: 150
- batch_size: 100
- Optimizer: SGD. Learning Rate: 0.01
- Data is not shuffled at each iteration

**Opened data located in data/TypeA**

- Num classes: 2
- Total Samples: 20000
- Vector size: 128
- Train percentage: 80%

**Models:**

**Complex Network**

Dense layer:

- input size = 128(<class 'numpy.complex64'>) -> output size = 100;
- act_fun = cart_relu;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: 0.5

Dense layer:

- input size = 100(complex64) -> output size = 40;
- act_fun = cart_relu;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: 0.5

Dense layer:

- input size = 40(complex64) -> output size = 2;
- act_fun = softmax_real;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: None

**Real Network**

Dense layer:

- input size = 256(<class 'numpy.float32'>) -> output size = 200;
- act_fun = cart_relu;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: 0.5

Dense layer:

- input size = 200(<class 'numpy.float32'>) -> output size = 80;
- act_fun = cart_relu;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: 0.5

Dense layer:

- input size = 80(<class 'numpy.float32'>) -> output size = 2;
- act_fun = softmax_real;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: None

Results Graphs
""""""""""""""

**Box Plots**

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/base_case_type_A_2HL_test_accuracy_box_plot.html

**Confidence lines**

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/base_case_type_A_2HL_test_loss.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/base_case_type_A_2HL_test_accuracy.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/montecarlo_train_loss.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/montecarlo_train_accuracy.html

**Histograms**

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/montecarlo_test_accuracy_histogram.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/montecarlo_test_loss_histogram.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/montecarlo_train_accuracy_histogram.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_a_2hl/montecarlo_train_loss_histogram.html