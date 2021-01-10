MNIST Example
-------------

This example is based on `Training a neural network on MNIST with Keras <https://www.tensorflow.org/datasets/keras_example>`_ and is used to help prove the correct performance of our model (as it renders the same result).

The code to test on MNIST is available on GitHub within `examples/mnist_dataset.py <https://github.com/NEGU93/cvnn/blob/master/examples/mnist_dataset.py>`_

First lets import whats needed::

    import tensorflow.compat.v2 as tf
    import tensorflow_datasets as tfds
    from cvnn import layers
    import numpy as np

    tfds.disable_progress_bar()
    tf.enable_v2_behavior()

Load MNIST dataset::

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    ds_train = ds_train.map(normaconda lize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

Create and train the model::

    model = tf.keras.models.Sequential([    # Remember to cast the dtype to float32
        layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32),
        layers.ComplexDense(128, activation='cart_relu', dtype=np.float32),
        layers.ComplexDense(10, activation='softmax_real', dtype=np.float32)
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    model.fit(ds_train, epochs=6, validation_data=ds_test, verbose=verbose, shuffle=False)

Finally, this code will render the following output::

    Epoch 1/6
    469/469 [==============================] - 6s 12ms/step - loss: 1.2619 - accuracy: 0.7003 - val_loss: 0.6821 - val_accuracy: 0.8506
    Epoch 2/6
    469/469 [==============================] - 8s 17ms/step - loss: 0.5765 - accuracy: 0.8602 - val_loss: 0.4727 - val_accuracy: 0.8802
    Epoch 3/6
    469/469 [==============================] - 7s 14ms/step - loss: 0.4525 - accuracy: 0.8816 - val_loss: 0.4023 - val_accuracy: 0.8964
    Epoch 4/6
    469/469 [==============================] - 5s 11ms/step - loss: 0.4003 - accuracy: 0.8916 - val_loss: 0.3657 - val_accuracy: 0.9024
    Epoch 5/6
    469/469 [==============================] - 6s 12ms/step - loss: 0.3696 - accuracy: 0.8983 - val_loss: 0.3418 - val_accuracy: 0.9071
    Epoch 6/6
    469/469 [==============================] - 5s 10ms/step - loss: 0.3488 - accuracy: 0.9024 - val_loss: 0.3267 - val_accuracy: 0.9112

**Statistical Results**

To assert the code works correctly, we have done 1000 iterations of both cvnn model and Keras model. The following box plot shows the results.

.. warning:: 
    ATTENTION: Accuracy is lower than in `Training a neural network on MNIST with Keras <https://www.tensorflow.org/datasets/keras_example>`_ because the optimizer used here is SGD and not Adam. Should we use SGD on the Keras example it will arrive to the same result.

.. raw:: html
   :file: ../source/_static/SGD_mnist_test.html
