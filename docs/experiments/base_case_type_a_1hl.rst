Base Case Type A 1HL
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

Dense layer

- input size = 128(<class 'numpy.complex64'>) -> output size = 64;
- act_fun = cart_relu;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: 0.5

Dense layer

- input size = 64(complex64) -> output size = 2;
- act_fun = softmax_real;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: None

**Real Network**

Dense layer

- input size = 256(<class 'numpy.float32'>) -> output size = 128;
- act_fun = cart_relu;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: 0.5

Dense layer

- input size = 128(<class 'numpy.float32'>) -> output size = 2;
- act_fun = softmax_real;
- weight init = Glorot Uniform; bias init = Zeros
- Dropout: None

Results Graphs
""""""""""""""

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_A_1HL_test_accuracy_box_plot.html

.. raw:: html
   :file: ../source/_static/circularity/base_case_type_A_1HL_test_accuracy_histogram.html