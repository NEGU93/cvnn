Several correlation coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Results Graph
"""""""""""""

.. raw:: html

    <iframe src="../_static/circularity/coef_correl_1HL_w_dropout.html" height="500px" width="100%"></iframe>



**Monte Carlo run**

- Iterations: 30
- epochs: 300
- batch_size: 100
- Optimizer: SGD. Learning Rate: 0.1
- Data is not shuffled at each iteration

**Opened data located in data/TypeA**

Correlation coefficient was changed from 0.1 to 0.9 in order to create the graph.

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