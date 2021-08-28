import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cvnn.layers as complex_layers
import numpy as np
from pdb import set_trace

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype(dtype=np.float32) / 255.0, test_images.astype(dtype=np.float32) / 255.0


def keras_fit(epochs=10, use_bias=True):
    tf.random.set_seed(1)
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_initializer=init,
                            use_bias=use_bias))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init, use_bias=use_bias))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init, use_bias=use_bias))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer=init, use_bias=use_bias))
    model.add(layers.Dense(10, kernel_initializer=init, use_bias=use_bias))
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    weigths = model.get_weights()
    with tf.GradientTape() as tape:
        # for elem, label in iter(ds_train):
        loss = model.compiled_loss(y_true=tf.convert_to_tensor(test_labels), y_pred=model(test_images))
        gradients = tape.gradient(loss, model.trainable_weights)  # back-propagation
    logs = {
        'weights': weigths,
        'loss': loss,
        'gradients': gradients
    }
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return history, logs


def own_fit(epochs=10):
    tf.random.set_seed(1)
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3),
                                           dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), dtype=np.float32))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.float32,
                                           kernel_initializer=init))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2), dtype=np.float32))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', dtype=np.float32,
                                           kernel_initializer=init))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_relu', dtype=np.float32, kernel_initializer=init))
    model.add(complex_layers.ComplexDense(10, dtype=np.float32, kernel_initializer=init))
    # model.summary()
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return history


def own_complex_fit(epochs=10):
    tf.random.set_seed(1)
    init = tf.keras.initializers.GlorotUniform(seed=117)
    model = models.Sequential()
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu', input_shape=(32, 32, 3),
                                           kernel_initializer=init, use_bias=False, init_technique='zero_imag'))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', kernel_initializer=init,
                                           use_bias=False, init_technique='zero_imag'))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu', kernel_initializer=init,
                                           use_bias=False, init_technique='zero_imag'))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_relu', kernel_initializer=init,
                                          use_bias=False, init_technique='zero_imag'))
    model.add(complex_layers.ComplexDense(10, activation='cast_to_real', kernel_initializer=init,
                                          use_bias=False, init_technique='zero_imag'))
    # model.summary()
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    weigths = model.get_weights()
    with tf.GradientTape() as tape:
        loss = model.compiled_loss(y_true=tf.convert_to_tensor(test_labels), y_pred=model(test_images))
        gradients = tape.gradient(loss, model.trainable_weights)  # back-propagation
    logs = {
        'weights': weigths,
        'loss': loss,
        'gradients': gradients
    }
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return history, logs


def test_cifar10():
    epochs = 3
    assert not tf.test.gpu_device_name(), "Using GPU not good for debugging"
    keras, keras_logs = keras_fit(epochs=epochs, use_bias=False)
    # keras1 = keras_fit(epochs=epochs)
    own, own_logs = own_complex_fit(epochs=epochs)
    set_trace()
    assert np.all([np.all(k_w == o_w) for k_w, o_w in zip(keras_logs['weights'], own_logs['weights'][::2])])
    assert np.all([np.all(o_w == 0) for o_w in own_logs['weights'][1::2]])
    assert own_logs['loss'] == keras_logs['loss']
    assert np.all([np.all(k == o) for k, o in zip(keras_logs['weights'], own_logs['weights'][::2])])
    # TODO: This is indeed strange, gradients are the same! Not even similar, but result is not.
    # assert keras.history == own.history, f"\n{keras.history}\n !=\n{own.history}"

    keras, _ = keras_fit(epochs=epochs)
    # keras1 = keras_fit(epochs=epochs)
    own = own_fit(epochs=epochs)
    assert keras.history == own.history, f"\n{keras.history}\n !=\n{own.history}"
    # for k, k2, o in zip(keras.history.values(), keras1.history.values(), own.history.values()):
    #     if np.all(np.array(k) == np.array(k2)):
    #         assert np.all(np.array(k) == np.array(o)), f"\n{keras.history}\n !=\n{own.history}"


if __name__ == "__main__":
    from importlib import reload
    import os
    import tensorflow
    reload(tensorflow)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    test_cifar10()
