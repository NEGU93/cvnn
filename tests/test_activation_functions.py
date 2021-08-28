import tensorflow as tf
from cvnn import layers, activations

if __name__ == '__main__':
    for activation in activations.act_dispatcher.keys():
        print(activation)
        model = tf.keras.Sequential([
            layers.ComplexInput(4),
            layers.ComplexDense(1, activation=activation),
            layers.ComplexDense(1, activation='linear')
        ])