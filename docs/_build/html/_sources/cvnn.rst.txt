CVNN
===========



Short example::

        # Assume you already have complex data 'x' with its labels 'y'...

        shape = [layers.ComplexDense(output_size=100, input_size=np.shape(x)[1], activation='cart_relu'),
                layers.ComplexDense(output_size=40, activation='cart_relu'),
                layers.ComplexDense(output_size=np.shape(y)[1], activation='softmax_real')]
        model = CvnnModel("cvnn_example", shape, tf.keras.losses.categorical_crossentropy)
        model.fit(x, y, batch_size=100, epochs=150)

