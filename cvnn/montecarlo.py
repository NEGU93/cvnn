import cvnn.layers as layers
import cvnn.dataset as dp
from cvnn.dataset import Dataset
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
        self.monte_carlo_analyzer = MonteCarloAnalyzer()  # All at None

    def add_model(self, model):
        assert model.save_csv_checkpoints
        self.models.append(model)

    def run(self, x, y, data_summary='',
            iterations=100, learning_rate=0.01, epochs=10, batch_size=100, shuffle=True, debug=False, display_freq=160):
        self.pandas_full_data = pd.DataFrame()  # Reset data frame
        self.save_summary_of_run(self._run_summary(iterations, learning_rate, epochs, batch_size, shuffle),
                                 data_summary)
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
                               verbose=debug, fast_mode=not debug, save_to_file=False, display_freq=display_freq)
                self.pandas_full_data = pd.concat([self.pandas_full_data,
                                                   test_model.plotter.get_full_pandas_dataframe()])
            # Save checkpoint in case montecarlo stops in the middle
            self.pandas_full_data.to_csv(self.monte_carlo_analyzer.path / "checkpoints/iteration{}.csv".format(it + 1),
                                         index=False)
        self.pandas_full_data = self.pandas_full_data.reset_index(drop=True)
        self.monte_carlo_analyzer.set_df(self.pandas_full_data)

    @staticmethod
    def _run_summary(iterations, learning_rate, epochs, batch_size, shuffle):
        ret_str = "Monte Carlo run\n"
        ret_str += "\tIterations: {}\n".format(iterations)
        ret_str += "\tepochs: {}\n".format(epochs)
        ret_str += "\tbatch_size: {}\n".format(batch_size)
        ret_str += "\tLearning Rate: {}\n".format(learning_rate)
        if shuffle:
            ret_str += "\tShuffle data at each iteration\n"
        else:
            ret_str += "\tData is not shuffled at each iteration\n"
        return ret_str

    def save_summary_of_run(self, run_summary, data_summary):
        with open(str(self.monte_carlo_analyzer.path / "run_summary.txt"), "w") as file:
            file.write(run_summary)
            file.write(data_summary)
            file.write("Models:\n")
            for model in self.models:
                file.write(model.summary())


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


def run_montecarlo(iterations=1000, m=10000, n=128, num_classes=2, coeff_correl_limit=0.75,
                   epochs=150, batch_size=100, display_freq=None, learning_rate=0.002,
                   shape_raw=None, activation='cart_relu', debug=False):
    if shape_raw is None:
        shape_raw = [100, 40]
    if display_freq is None:
        display_freq = int(m*num_classes*0.8/batch_size)
    dataset = dp.CorrelatedGaussianNormal(m, n, num_classes, debug=False, coeff_correl_limit=coeff_correl_limit)
    x, y = dataset.get_all()
    x = x.astype(np.complex64)
    y = dp.Dataset.sparse_into_categorical(y, num_classes=num_classes).astype(np.float32)
    x_train, y_train, x_test, y_test = dp.Dataset.separate_into_train_and_test(x, y)
    x_train_real = transform_to_real(x_train)
    x_test_real = transform_to_real(x_test)

    # Create complex network
    input_size = x.shape[1]  # Size of input
    output_size = y.shape[1]  # Size of output
    assert len(shape_raw) > 0

    shape = [ComplexDense(input_size=input_size, output_size=shape_raw[0], activation=activation,
                          input_dtype=np.complex64, output_dtype=np.complex64)]
    for i in range(1, len(shape_raw)):
        shape.append(ComplexDense(input_size=shape_raw[i - 1], output_size=shape_raw[i], activation=activation,
                                  input_dtype=np.complex64, output_dtype=np.complex64))
    shape.append(ComplexDense(input_size=shape_raw[-1], output_size=output_size, activation='cart_softmax_real',
                              input_dtype=np.complex64, output_dtype=np.float32))

    complex_network = CvnnModel(name="complex_network", shape=shape, loss_fun=tf.keras.losses.categorical_crossentropy,
                                verbose=False, tensorboard=False, save_csv_checkpoints=True)

    # Monte Carlo
    monte_carlo = RealVsComplex(complex_network)
    monte_carlo.run(x, y, iterations=iterations, learning_rate=learning_rate,
                    epochs=epochs, batch_size=batch_size, display_freq=display_freq,
                    shuffle=True, debug=debug, data_summary=dataset.summary())


if __name__ == "__main__":
    # testing
    run_montecarlo(iterations=5, epochs=10, debug=True)

    # Run the base case
    run_montecarlo()

    # change pearson coefficient
    coefs = np.linspace(0, 0.999, 11)[1:]
    for coef in coefs:
        run_montecarlo(coeff_correl_limit=coef)

    # change m
    per_class_examples = [5000, 2000, 1000, 500]
    for m in per_class_examples:
        run_montecarlo(m=m)

    # change learning rate
    learning_rates = [0.001, 0.01, 0.1]
    for learning_rate in learning_rates:
        run_montecarlo(learning_rate=learning_rate)

    # change activation function
    activation_function = ['cart_sigmoid', 'cart_tanh']
    for activation in activation_function:
        run_montecarlo(activation=activation)

    shapes = [
        [128],
        [256],
        [64],
        [32],
        [64, 32],
        [128, 32],
        [128, 64, 32]
    ]
    for shape in shapes:
        run_montecarlo(shape_raw=shape)
