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

    def run(self, x, y, data_summary='', polar=False,
            iterations=100, learning_rate=0.01, epochs=10, batch_size=100, shuffle=False, debug=False, display_freq=160):
        self.pandas_full_data = pd.DataFrame()  # Reset data frame
        self.save_summary_of_run(self._run_summary(iterations, learning_rate, epochs, batch_size, shuffle),
                                 data_summary)
        os.makedirs(self.monte_carlo_analyzer.path / "checkpoints/", exist_ok=True)
        for it in range(iterations):
            print("Iteration {}/{}".format(it + 1, iterations))
            if shuffle:     # shuffle all data at each iteration
                x, y = randomize(x, y)
            for i, model in enumerate(self.models):
                if model.is_complex():
                    x_fit = x
                else:
                    x_fit = transform_to_real(x, polar=polar)
                test_model = copy.deepcopy(model)
                test_model.fit(x_fit, y,
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


def run_montecarlo(iterations=1000, m=10000, n=128, param_list=None,
                   epochs=150, batch_size=100, display_freq=None, learning_rate=0.01,
                   shape_raw=None, activation='cart_relu', debug=False, polar=False):
    # Get parameters
    if shape_raw is None:
        shape_raw = [100, 40]
    if param_list is None:
        param_list = [
            [0.5, 1, 2],
            [-0.5, 1, 2]
        ]
    if display_freq is None:
        display_freq = int(m * len(param_list) * 0.8 / batch_size)
    dataset = dp.CorrelatedGaussianCoeffCorrel(m, n, param_list, debug=False)

    # Create complex network
    input_size = dataset.x.shape[1]  # Size of input
    output_size = dataset.y.shape[1]  # Size of output
    assert len(shape_raw) > 0
    shape = [ComplexDense(input_size=input_size, output_size=shape_raw[0], activation=activation,
                          input_dtype=np.complex64, output_dtype=np.complex64)]
    for i in range(1, len(shape_raw)):
        shape.append(ComplexDense(input_size=shape_raw[i - 1], output_size=shape_raw[i], activation=activation,
                                  input_dtype=np.complex64, output_dtype=np.complex64))
    shape.append(ComplexDense(input_size=shape_raw[-1], output_size=output_size, activation='softmax_real',
                              input_dtype=np.complex64, output_dtype=np.float32))

    complex_network = CvnnModel(name="complex_network", shape=shape, loss_fun=tf.keras.losses.categorical_crossentropy,
                                verbose=False, tensorboard=False, save_csv_checkpoints=True)

    # Monte Carlo
    monte_carlo = RealVsComplex(complex_network)
    monte_carlo.run(dataset.x, dataset.y, iterations=iterations, learning_rate=learning_rate,
                    epochs=epochs, batch_size=batch_size, display_freq=display_freq,
                    shuffle=True, debug=debug, data_summary=dataset.summary(), polar=polar)


def first_simus():
    print("Running base case monte carlo")
    run_montecarlo()

    # Equal variance
    print("Running monte carlo with equal variance")
    param_list = [
        [0.5, 1, 1],
        [-0.5, 1, 1]
    ]
    run_montecarlo(param_list=param_list)

    # No correlation
    print("Running monte carlo with no correlation")
    param_list = [[0, 1, 2], [0, 2, 1]]
    run_montecarlo(param_list=param_list)  # I have the theory polar will do bad here...

    # Multi Class
    print("Running multi-class (4) monte carlo ")
    coef_correls_list = np.linspace(-0.9, 0.9, 4)  # 4 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 2])
    run_montecarlo(param_list=param_list)
    print("Running multi-class (10) monte carlo ")
    coef_correls_list = np.linspace(-0.9, 0.9, 10)  # 10 classes
    param_list = []
    for coef in coef_correls_list:
        param_list.append([coef, 1, 2])
    run_montecarlo(param_list=param_list)

    # change activation function
    print("Running monte carlo for different activation functions")
    activation_function = ['cart_sigmoid', 'cart_tanh', 'cart_leaky_relu']  # relu already done
    for activation in activation_function:
        run_montecarlo(activation=activation)

    # change m
    print("Running monte carlo for different sizes")
    per_class_examples = [5000, 2000, 1000, 500]  # 10000 already done
    for m in per_class_examples:
        run_montecarlo(m=m)

    # change learning rate TODO: RUN AGAIN!
    learning_rates = [0.001, 0.1, 1]        # 0.01 already done
    for learning_rate in learning_rates:
        run_montecarlo(learning_rate=learning_rate)


def shapes_simus():
    # Single hidden layers
    shapes = [
        [512],
        [256],
        [128],
        [64],
        [32],
        [16],
        [8]
    ]
    for shape in shapes:
        print("Running for shape {}".format(shape[0]))
        run_montecarlo(shape_raw=shape)

    # Two hidden layers
    shapes = [
        [512, 100],
        [256, 50],
        [85, 40],
        [64, 32],
        [32, 50],
        [32, 80]
    ]
    for shape in shapes:
        print("Running for 2 hidden layers".format(shape[0]))
        run_montecarlo(shape_raw=shape)


if __name__ == "__main__":
    print("Running base case monte carlo")
    run_montecarlo(polar=True)

    # Equal variance
    print("Running monte carlo with equal variance")
    param_list = [
        [0.5, 1, 1],
        [-0.5, 1, 1]
    ]
    run_montecarlo(param_list=param_list, polar=True)

    # No correlation
    print("Running monte carlo with no correlation")
    param_list = [[0, 1, 2], [0, 2, 1]]
    run_montecarlo(param_list=param_list, polar=True)  # I have the theory polar will do bad here...
