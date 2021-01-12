CVNN
===========

This shows a simple example on how to use this library.

First lets import what we need


.. code-block:: python

        import numpy as np
        import cvnn.layers as complex_layers
        import tensorflow as tf

We will assume you have a `get_dataset()` function that has it's complex dtype data. 

If you don't yet have it and you want first to test any working example, you can use the following code.

.. code-block:: python

        def get_dataset():
                (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
                train_images = train_images.astype(dtype=np.complex64) / 255.0
                test_images = test_images.astype(dtype=np.complex64) / 255.0
                return (train_images, train_labels), (test_images, test_labels)

.. warning::
        This will just make it have a nul imaginary part (:code:`z = x + 1j*0`), it makes no sense to use a complex network for this dataset. It is just for an example.

Ok, we are now ready to create our model! Let's create a Complex-Valued Convolutional Neural Netowrk (CV-CNN)

.. warning::
        Tensorflow casts the input automatically to real! To avoid that, use always the ComplexInput layer as the input.

.. code-block:: python

        # Assume you already have complex data... example numpy arrays of dtype np.complex64
        (train_images, train_labels), (test_images, test_labels) = get_dataset()        # to be done by each user

        model = tf.keras.models.Sequential()
        model.add(complex_layers.ComplexInput(input_shape=(32, 32, 3)))                     # Always use ComplexInput at the start
        model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
        model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
        model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
        model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
        model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
        model.add(complex_layers.ComplexFlatten())
        model.add(complex_layers.ComplexDense(64, activation='cart_relu'))
        model.add(complex_layers.ComplexDense(10, activation='convert_to_real_with_abs')) 
        model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        model.summary() 

.. note::
        An activation that casts to real must be used at the last layer as the loss function cannot minimize a complex number.

The last code will output the model summary::

        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        complex_conv2d (ComplexConv2 (None, 30, 30, 32)        1792      
        _________________________________________________________________
        complex_avg_pooling2d (Compl (None, 15, 15, 32)        0         
        _________________________________________________________________
        complex_conv2d_1 (ComplexCon (None, 13, 13, 64)        36992     
        _________________________________________________________________
        complex_flatten (ComplexFlat (None, 10816)             0         
        _________________________________________________________________
        complex_dense (ComplexDense) (None, 64)                1384576   
        _________________________________________________________________
        complex_dense_1 (ComplexDens (None, 10)                1300      
        =================================================================
        Total params: 1,424,660
        Trainable params: 1,424,660
        Non-trainable params: 0
        _________________________________________________________________

Great! we have our model done, now we are ready to train!

.. code-block:: python 

        history = model.fit(train_images, train_labels, epochs=6, validation_data=(test_images, test_labels))

Training output::

        Epoch 1/6
        1563/1563 [==============================] - 21s 13ms/step - loss: 1.4151 - accuracy: 0.4932 - val_loss: 1.1865 - val_accuracy: 0.5854
        Epoch 2/6
        1563/1563 [==============================] - 17s 11ms/step - loss: 1.0378 - accuracy: 0.6339 - val_loss: 1.0505 - val_accuracy: 0.6415
        Epoch 3/6
        1563/1563 [==============================] - 17s 11ms/step - loss: 0.8625 - accuracy: 0.6968 - val_loss: 0.9945 - val_accuracy: 0.6575
        Epoch 4/6
        1563/1563 [==============================] - 15s 10ms/step - loss: 0.7133 - accuracy: 0.7499 - val_loss: 0.9414 - val_accuracy: 0.6774
        Epoch 5/6
        1563/1563 [==============================] - 16s 11ms/step - loss: 0.5716 - accuracy: 0.7999 - val_loss: 0.9673 - val_accuracy: 0.6895
        Epoch 6/6
        1563/1563 [==============================] - 18s 11ms/step - loss: 0.4350 - accuracy: 0.8490 - val_loss: 1.0668 - val_accuracy: 0.6848
        
To evaluate the models performance you can use

.. code-block:: python

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

Output::

        313/313 - 2s - loss: 1.0668 - accuracy: 0.6848

You can now predict using either :code:`model(test_images)` or :code:`model.predict(test_images)`.