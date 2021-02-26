import tensorflow as tf
import numpy as np
import cvnn.layers as complex_layers


def all_layers_model():
    """
    Creates a model using all possible layers to assert no layer changes the dtype to real.
    """
    input_shape = (4, 28, 28, 3)
    x = tf.cast(tf.random.normal(input_shape), tf.complex64)

    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=input_shape[1:]))  # Always use ComplexInput at the start
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_sigmoid'))
    model.add(complex_layers.ComplexDropout(0.5))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_tanh'))
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer='adam', metrics=['accuracy'])
    y = model(x)
    assert y.dtype == np.complex64
    return model


def test_output_dtype():
    model = all_layers_model()


if __name__ == "__main__":
    test_output_dtype()
