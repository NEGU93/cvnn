import cvnn.layers as layers
import cvnn.data_processing as dp
from cvnn.cvnn_model import CvnnModel
from datetime import datetime
from pathlib import Path
import sys
import os
import numpy as np
from pdb import set_trace


class MonteCarlo:

    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def run(self, x, y, filename='monecarlo_run',
            iterations=100, learning_rate=0.01, epochs=10, batch_size=100, shuffle=False, debug=False):
        x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(x, y)
        x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)
        x_train_real = x_train_real.astype(np.float32)
        x_test_real = x_test_real.astype(np.float32)
        now = datetime.today()
        path = Path("./results/" + now.strftime("%Y/%m%B/%d%A/run-%Hh%Mm%S/"))
        if not os.path.exists(path):
            os.makedirs(path)
        filename = str(path / filename)
        if not filename.endswith(".csv"):
            filename += ".csv"
        if os.path.exists(filename):
            exit("File already existed, aborting to prevent override")
        file = open(filename, 'a')
        print("Writing results into " + filename)
        for i, model in enumerate(self.models):
            if i != 0:
                file.write(",")
            file.write("{0} train loss,{0} train acc,{0} test loss,{0} test acc".format(model.name))
        file.write("\n")
        for it in range(iterations):
            print("Iteration {}/{}".format(it + 1, iterations))
            if shuffle:
                x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(x, y)
                x_train_real, x_test_real = dp.separate_into_train_and_test(x_train, x_test)
            for i, model in enumerate(self.models):
                if model.is_complex():
                    x_train_iter = x_train
                    x_val = x_test
                else:
                    x_train_iter = x_train_real
                    x_val = x_test_real
                test_shape = []
                for layer in model.shape:
                    if isinstance(layer, layers.ComplexDense):
                        test_shape.append(layers.ComplexDense(layer.input_size, layer.output_size,
                                                              activation=layer.activation,
                                                              input_dtype=layer.input_dtype,
                                                              output_dtype=layer.output_dtype,
                                                              weight_initializer=layer.weight_initializer,
                                                              bias_initializer=layer.bias_initializer
                                                              ))
                    else:
                        sys.exit("Layer " + str(layer) + " unknown")
                test_model = CvnnModel(model.name, test_shape, model.loss_fun,
                                       verbose=debug, tensorboard=model.tensorboard,
                                       save_model_checkpoints=False, save_csv_checkpoints=model.save_csv_checkpoints)
                test_model.fit(x_train_iter, y_train, x_test=x_val, y_test=y_test,
                               learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                               verbose=debug, fast_mode=not debug, save_to_file=False)
                train_loss, train_acc = test_model.evaluate(x_train_iter, y_train)
                test_loss, test_acc = test_model.evaluate(x_val, y_test)
                # save result
                file.write(str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc))
                if i != len(self.models) - 1:
                    file.write(",")
                else:
                    file.write("\n")
                    file.flush()  # Not to lose the data if MC stops in the middle
                    # typically the above line would do. however this is used to ensure that the file is written
                    os.fsync(file.fileno())  # http://docs.python.org/2/library/stdtypes.html#file.flush
        file.close()


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
