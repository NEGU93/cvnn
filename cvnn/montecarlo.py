import cvnn.layers as layers
import cvnn.data_processing as dp
from cvnn.data_processing import Dataset
from cvnn.cvnn_model import CvnnModel
from cvnn.data_analysis import MonteCarloAnalyzer
from cvnn.layers import ComplexDense
from utils import create_folder, transform_to_real, randomize
import tensorflow as tf
import pandas as pd
import copy
import sys
import os
import numpy as np
from pdb import set_trace


class MonteCarlo:

    def __init__(self):
        self.models = []
        self.pandas_full_data = pd.DataFrame()
        self.monte_carlo_analyzer = MonteCarloAnalyzer()      # All at None

    def add_model(self, model):
        assert model.save_csv_checkpoints
        self.models.append(model)

    def run(self, x, y,
            iterations=100, learning_rate=0.01, epochs=10, batch_size=100, shuffle=True, debug=False):
        self.pandas_full_data = pd.DataFrame()      # Reset data frame
        x_train, y_train, x_val, y_val = Dataset.separate_into_train_and_test(x, y)
        x_train_real = transform_to_real(x_train)
        x_test_real = transform_to_real(x_val)
        if not os.path.exists(self.monte_carlo_analyzer.path / "checkpoints/"):
            os.makedirs(self.monte_carlo_analyzer.path / "checkpoints/")
        for it in range(iterations):
            print("Iteration {}/{}".format(it + 1, iterations))
            if shuffle:
                x, y = randomize(x, y)
                x_train, y_train, x_val, y_val = Dataset.separate_into_train_and_test(x, y)
                x_train_real = transform_to_real(x_train)
                x_test_real = transform_to_real(x_val)
            for i, model in enumerate(self.models):
                if model.is_complex():
                    x_train_iter = x_train
                    x_val = x_val
                else:
                    x_train_iter = x_train_real
                    x_val = x_test_real
                test_model = copy.deepcopy(model)
                test_model.fit(x_train_iter, y_train, x_test=x_val, y_test=y_val,
                               learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                               verbose=debug, fast_mode=not debug, save_to_file=False)
                self.pandas_full_data = pd.concat([self.pandas_full_data,
                                                   test_model.plotter.get_full_pandas_dataframe()])
            # Save checkpoint in case montecarlo stops in the middle
            self.pandas_full_data.to_csv(self.monte_carlo_analyzer.path / "checkpoints/iteration{}.csv".format(it + 1), index=False)
        self.pandas_full_data = self.pandas_full_data.reset_index(drop=True)
        self.monte_carlo_analyzer.set_df(self.pandas_full_data)


class RealVsComplex(MonteCarlo):

    def __init__(self, complex_model):
        super().__init__()
        # generate real network shape
        real_shape = []
        output_mult = 2
        for i, layer in enumerate(complex_model.shape):
            if i == len(complex_model.shape) - 1:
                output_mult = 1  # Do not multiply last layer
            # Do all the supported layers
            if isinstance(layer, layers.ComplexDense):
                real_shape.append(layers.ComplexDense(layer.input_size * 2, layer.output_size * output_mult,
                                                      activation=layer.activation,
                                                      input_dtype=np.float32, output_dtype=np.float32,
                                                      weight_initializer=layer.weight_initializer,
                                                      bias_initializer=layer.bias_initializer
                                                      ))
            else:
                sys.exit("Layer " + str(layer) + " unknown")
        # add models
        self.add_model(complex_model)
        self.add_model(CvnnModel(name="real_network", shape=real_shape, loss_fun=complex_model.loss_fun,
                                 tensorboard=complex_model.tensorboard, verbose=False,
                                 save_model_checkpoints=complex_model.save_model_checkpoints,
                                 save_csv_checkpoints=complex_model.save_csv_checkpoints))


if __name__ == "__main__":
    m = 10000
    n = 100
    num_classes = 2
    dataset = dp.CorrelatedGaussianNormal(m, n, num_classes, debug=False, coeff_correl_limit=0.75)
    x, y = dataset.get_all()
    x = x.astype(np.complex64)
    y = dp.Dataset.sparse_into_categorical(y, num_classes=num_classes).astype(np.float32)
    x_train, y_train, x_test, y_test = dp.Dataset.separate_into_train_and_test(x, y)
    x_train_real = transform_to_real(x_train)
    x_test_real = transform_to_real(x_test)

    epochs = 100
    batch_size = 100
    display_freq = 160
    learning_rate = 0.002

    # Create complex network
    input_size = x.shape[1]  # Size of input
    output_size = y.shape[1]  # Size of output
    h1_size = 100
    h2_size = 40
    shape = [ComplexDense(input_size=input_size, output_size=h1_size, activation='cart_relu',
                          input_dtype=np.complex64, output_dtype=np.complex64),
             ComplexDense(input_size=h1_size, output_size=h2_size, activation='cart_relu',
                          input_dtype=np.complex64, output_dtype=np.complex64),
             ComplexDense(input_size=h2_size, output_size=output_size, activation='cart_softmax_real',
                          input_dtype=np.complex64, output_dtype=np.float32)]
    complex_network = CvnnModel(name="complex_network", shape=shape, loss_fun=tf.keras.losses.categorical_crossentropy,
                                verbose=False, tensorboard=False, save_csv_checkpoints=True)
    # Monte Carlo
    monte_carlo = RealVsComplex(complex_network)
    monte_carlo.run(x, y, iterations=100, learning_rate=learning_rate,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True, debug=False)
    monte_carlo.monte_carlo_analyzer.plot_histogram(library='plotly')


