Complex Dense
-------------

.. py:class:: ComplexDense

    Fully connected complex-valued layer.

    Implements the operation:

    .. math::

        \sigma(\textrm{input * weights + bias}) 

    * where data types can be either complex or real.
    * activation (:math:`\sigma`) is the element-wise activation function passed as the activation argument, 
    * weights is a matrix created by the layer
    * bias is a bias vector created by the layer

.. py:method:: __init__(self, units, activation=None, use_bias=True, kernel_initializer=ComplexGlorotUniform(), bias_initializer=Zeros(), dtype=DEFAULT_COMPLEX_TYPE, init_technique: str = 'mirror', **kwargs)

        Initializer of the Dense layer

        :param units: Positive integer, dimensionality of the output space.
        :param activation: Activation function to use. 
            Either from :code:`keras.activations` or :code:`cvnn.activations`. For complex dtype, only :code:`cvnn.activations` module supported.
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param kernel_initializer: Initializer for the kernel weights matrix.
            Recomended to use a :code:`ComplexInitializer` such as :code:`cvnn.initializers.ComplexGlorotUniform()` (default)
        :param bias_initializer: Initializer for the bias vector.
            Recomended to use a :code:`ComplexInitializer` such as :code:`cvnn.initializers.Zeros()` (default)
        :param dtype: Dtype of the input and layer.
        :param init_technique: String. One of 'mirror' or 'zero_imag'. Tells the initializer how to init complex number if the initializer was tensorflow's built in initializers (not supporting complex numbers).
            
            - 'mirror' (default): Uses the initializer for both real and imaginary part. Note that some initializers such as Glorot or He will lose it's property if initialized this way.
            - 'zero_imag': Initializer real part and let imaginary part to zero.

**Code example**

Let's first get some data to test

.. code-block:: python

    img_r = np.array([[
        [0, 1, 2],
        [0, 2, 2],
        [0, 5, 7]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img_i = np.array([[
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ], [
        [0, 4, 5],
        [3, 7, 9],
        [4, 5, 3]
    ]]).astype(np.float32)
    img = img_r + 1j * img_i

Ok, we are now ready to run it through a :code:`ComplexDense` layer (with a :code:`ComplexFlatten` first of couse)

.. code-block:: python

    c_flat = ComplexFlatten()
    c_dense = ComplexDense(units=10)
    res = c_dense(c_flat(img.astype(np.complex64)))
    assert res.shape == [2, 10]
    assert res.dtype == tf.complex64 

Now, to do this in a :code:`Sequential` model we can do it like:

.. code-block:: python

    model = tf.keras.models.Sequential()
    model.add(ComplexInput(input_shape=(3, 3)))
    model.add(ComplexFlatten())
    model.add(ComplexDense(32, activation='cart_relu'))
    model.add(ComplexDense(32))
    model.output_shape

This will output :code:`(None, 32)`. You can run the data created previously with

.. code-block:: python

    res = model(img.astype(np.complex64))
    assert res.dtype == tf.complex64
    
Doing now :code:`model.summary()` will output

.. code-block:: python

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    complex_flatten_1 (ComplexFl (None, 9)                 0
    _________________________________________________________________
    complex_dense_1 (ComplexDens (None, 32)                640
    _________________________________________________________________
    complex_dense_2 (ComplexDens (None, 32)                2112
    =================================================================
    Total params: 2,752
    Trainable params: 2,752
    Non-trainable params: 0
    _________________________________________________________________

.. note::

    If the input to the layer has a rank greater than 2, then Dense computes the dot product between the inputs and the kernel along the last axis of the inputs and axis 1 of the kernel (using :code:`tf.tensordot`). For example, if input has dimensions :code:`(batch_size, d0, d1)`, then we create a kernel with shape :code:`(d1, units)`, and the kernel operates along axis 2 of the input, on every sub-tensor of shape :code:`(1, 1, d1)` (there are batch_size * d0 such sub-tensors). The output in this case will have shape :code:`(batch_size, d0, units)`.
