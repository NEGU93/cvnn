import tensorflow as tf
import data_processing as dp
from utils import *
from datetime import datetime
from activation_functions import *
import numpy as np
import glob
import sys
import os
from pdb import set_trace

# https://ml-cheatsheet.readthedocs.io/en/latest/

DEBUG_RESTORE_META = False      # True to print the op and tensors that can be retrieved
DEBUG_WEIGHT_LOADER = True      # True to print the file being restored for the weights


def check_tf_version():
    # Makes sure Tensorflow version is 2
    assert tf.__version__.startswith('2')


def check_gpu_compatible():
    print("Available GPU devices:", flush=True)
    print(tf.test.gpu_device_name(), flush=True)
    print("Built in with CUDA: " + str(tf.test.is_built_with_cuda()), flush=True)
    print("GPU available: " + str(tf.test.is_gpu_available()), flush=True)


class Cvnn:
    """-------------------------
    # Constructor and Destructor
    -------------------------"""
    def __init__(self, name, learning_rate=0.001, tensorboard=True, verbose=True, automatic_restore=True):
        """
        Constructor
        :param name: Name of the network to be created. This will be used to save data into ./log/<name>/run-{date}/
        :param learning_rate: Learning rate at which the network will train TODO: this should not be here
        :param tensorboard: True if want the network to save tensorboard graph and summary
        :param verbose: True for verbose mode (print and output results)
        :param automatic_restore: True if network should search for saved models (will look for the newest saved model)
        """
        tf.compat.v1.disable_eager_execution()      # This class works as a graph model so no eager compatible
        # Save parameters of the constructor
        self.name = name
        self.verbose = verbose
        self.automatic_restore = automatic_restore
        self.tensorboard = tensorboard
        self.learning_rate = learning_rate

        # logs dir
        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_dir = "./log/{}/run-{}/".format(self.name, self.now)
        # Tensorboard
        self.tbdir = root_dir + "tensorboard_logs/"
        if not os.path.exists(self.tbdir):
            os.makedirs(self.tbdir)
        # checkpoint models
        self.savedir = root_dir + "saved_models/"
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # Launch the graph in a session.
        self.restored_meta = False
        if automatic_restore:
            self.restore_graph_from_meta()

        self._save_object_summary(root_dir)     # Save info to metadata

    def __del__(self):
        """
        Destructor
        :return: None
        """
        if self.tensorboard:
            try:
                self.writer.close()
            except AttributeError:
                print("Writer did not exist, couldn't delete it")
        try:        # TODO: better way to do it?
            self.sess.close()
        except AttributeError:
            print("Session was not created")

    """----------------
    # metadata.txt file
    ----------------"""
    def _save_object_summary(self, root_dir):
        """
        Create a .txt inside the root_dir with the information of this object in particular.
        If the file already exists it exits with a fatal message not to override information.
        :param root_dir: Directory path to where the txt file will be saved
        :return: None
        """
        try:
            self.metadata_filename = root_dir + "metadata.txt"
            with open(self.metadata_filename, "x") as file:
                # 'x' mode creates a new file. If file already exists, the operation fails
                file.write(self.name + "\n")
                file.write(self.now + "\n")
                file.write("automatic_restore, " + str(self.automatic_restore) + "\n")
                file.write("Restored," + str(self.restored_meta) + "\n")
                file.write("Tensorboard enabled, " + str(self.tensorboard) + "\n")
                file.write("Learning Rate, " + str(self.learning_rate) + "\n")
                file.write("Weight initialization, " + "uniform distribution over [0, 1)")   # TODO: change to correct
        except FileExistsError:     # TODO: Check if this is the actual error
            sys.error("Fatal: Same file already exists. Aborting to not override results")

    def _append_graph_structure(self, shape):
        """
        Appends the shape of the network to the metadata file.
        It checks the meta data file exists, if not throws and error and exits.
        :param shape: Shape of the network to be saved
        :return: None
        """
        if not os.path.exists(self.metadata_filename):
            sys.exit("Cvnn::_append_graph_structure: The meta data file did not exist!")
        with open(self.metadata_filename, "a") as file:
            # 'a' mode Opens a file for appending. If the file does not exist, it creates a new file for writing.
            file.write("\n")
            for i in range(len(shape)):
                if i == 0:
                    file.write("input layer, " + str(shape[i][0]))
                elif i == len(shape) - 1:
                    file.write("output layer, " + str(shape[i][0]))
                else:
                    file.write("hidden layer " + str(i) + ", " + str(shape[i][0]))
                if callable(shape[i][1]):           # Only write if the parameter was indeed a function
                    file.write(", " + shape[i][1].__name__)
                file.write("\n")

    """-----------------------
    #          Train 
    -----------------------"""
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=100, display_freq=1000, normal=False):
        """
        Performs the training of the neural network.
        If automatic_restore is True but not metadata was found,
            it will try to load the weights of the newest previously saved model.
        :param x_train: Training data of shape (<training examples>, <input_size>)
        :param y_train: Labels of the training data of shape (<training examples>, <output_size>)
        :param x_test: Test data to display accuracy at the end of shape (<test examples>, <input_size>)
        :param y_test: Test labels of shape (<test examples>, <output_size>)
        :param epochs: Total number of training epochs
        :param batch_size: Training batch size.
            If this number is bigger than the total amount of training examples will display an error
        :param display_freq: Display results frequency.
            The frequency will be for each (epoch * batch_size + iteration) / display_freq
        :return: None
        """
        if np.shape(x_train)[0] < batch_size:  # TODO: make this case work as well. Just display a warning
            sys.exit("Cvnn::train(): Batch size was bigger than total amount of examples")
        if normal:
            x_train = normalize(x_train)    # TODO: This normalize could be a bit different for each and be bad.
            x_test = normalize(x_test)
        with self.sess.as_default():
            assert tf.compat.v1.get_default_session() is self.sess
            self._init_weights()

            # Run validation at beginning
            self.print_validation_loss(0, x_test, y_test)
            num_tr_iter = int(len(y_train) / batch_size)        # Number of training iterations in each epoch
            for epoch in range(epochs):
                # Randomly shuffle the training data at the beginning of each epoch
                x_train, y_train = randomize(x_train, y_train)
                for iteration in range(num_tr_iter):
                    start = iteration * batch_size
                    end = (iteration + 1) * batch_size
                    x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                    # Run optimization op (backpropagation)
                    feed_dict_batch = {self.X: x_batch, self.y: y_batch}
                    self.sess.run(self.training_op, feed_dict=feed_dict_batch)
                    if (epoch * batch_size + iteration) % display_freq == 0:
                        self.run_checkpoint(epoch, num_tr_iter, iteration, feed_dict_batch)

            # Run validation at the end
            feed_dict_valid = {self.X: x_test, self.y: y_test}
            loss_valid = self.sess.run(self.loss, feed_dict=feed_dict_valid)
            self.print_validation_loss(epoch+1, x_test, y_test)
            self.save_model("final", "valid_loss", loss_valid)

    """------------------------
    # Predict models and result
    ------------------------"""
    def predict(self, x):
        """
        Runs a single feedforward computation
        :param x: Input of the network
        :return: Output of the network
        """
        # TODO: Check that x has the correct shape!
        with self.sess.as_default():
            # assert tf.compat.v1.get_default_session() is self.sess
            feed_dict_valid = {self.X: x}
            return self.y_out.eval(feed_dict=feed_dict_valid)

    # TODO: precision, recall, f1_score
    def compute_accuracy(self, x_test, y_test):
        y_prob = self.predict(x_test)
        y_prediction = np.argmax(y_prob, axis=1)
        y_labels = np.argmax(y_test, axis=1)
        acc = np.sum(y_prediction == y_labels) / len(y_labels)
        if self.verbose:
            print("Accuracy: {0:.2%}".format(acc))
        return acc

    def compute_loss(self, x, y):
        with self.sess.as_default():
            # assert tf.compat.v1.get_default_session() is self.sess
            feed_dict_valid = {self.X: x, self.y: y}
            return self.sess.run(self.loss, feed_dict=feed_dict_valid)

    """-------------
    # Graph creation
    -------------"""
    # Layers
    def _create_dense_layer(self, input_size, output_size, input, layer_number):
        with tf.compat.v1.name_scope("dense_layer_" + str(layer_number)) as scope:
            w = tf.Variable(self.glorot_uniform_init(input_size, output_size).astype(np.float32),
                            name="weights" + str(layer_number))
            b = tf.Variable(np.zeros(output_size).astype(np.float32), name="bias" + str(layer_number))
            if self.tensorboard:
                tf.compat.v1.summary.histogram('real_weight_' + str(layer_number), w)
            return tf.add(tf.matmul(input, w), b), [w, b]

    def _create_complex_dense_layer(self, input_size, output_size, input_of_layer, layer_number):
        # TODO: treat bias as a weight. It might optimize training (no add operation, only mult)
        with tf.compat.v1.name_scope("dense_layer_" + str(layer_number)) as scope:
            # Create weight matrix initialized randomely from N~(0, 0.01)
            w = tf.Variable(tf.complex(self.glorot_uniform_init(input_size, output_size).astype(np.float32),
                                       self.glorot_uniform_init(input_size, output_size).astype(np.float32)),
                            name="weights" + str(layer_number))
            b = tf.Variable(tf.complex(np.zeros(output_size).astype(np.float32),
                                       np.zeros(output_size).astype(np.float32)), name="bias" + str(layer_number))
            if self.tensorboard:
                tf.compat.v1.summary.histogram('real_weight_' + str(layer_number), tf.math.real(w))
                tf.compat.v1.summary.histogram('imag_weight_' + str(layer_number), tf.math.imag(w))
            return tf.add(tf.matmul(input_of_layer, w), b), [w, b]

    def _create_graph_from_shape(self, shape, input_dtype=np.complex64, output_dtype=np.float32):
        if len(shape) < 2:
            sys.exit("Cvnn::_create_graph_from_shape: shape should be at least of lenth 2")
        # Define placeholders
        self.X = tf.compat.v1.placeholder(tf.dtypes.as_dtype(input_dtype), shape=[None, shape[0][0]], name='X')
        self.y = tf.compat.v1.placeholder(tf.dtypes.as_dtype(output_dtype), shape=[None, shape[-1][0]], name='Y')

        variables = []
        with tf.compat.v1.name_scope("forward_phase") as scope:
            out = self.apply_activation(shape[0][1], self.X)
            for i in range(len(shape) - 1):  # Apply all the layers
                if input_dtype == np.complex64:
                    out, variable = self._create_complex_dense_layer(shape[i][0], shape[i + 1][0], out, i+1)
                elif input_dtype == np.float32:
                    out, variable = self._create_dense_layer(shape[i][0], shape[i + 1][0], out, i+1)
                else:   # TODO: add the rest of data types
                    sys.exit("CVNN::_create_graph_from_shape: input_type " + str(input_dtype) + " not supported")
                variables.extend(variable)
                out = self.apply_activation(shape[i + 1][1], out)           # Apply activation function
            y_out = tf.compat.v1.identity(out, name="y_out")
        if tf.dtypes.as_dtype(np.dtype(output_dtype)) != y_out.dtype:       # Case for real output / real labels
            y_out = tf.abs(y_out)       # TODO: Shall I do abs or what?
        self._append_graph_structure(shape)     # Append the graph information to the metadata.txt file
        return y_out, variables

    # Graphs
    def create_mlp_graph(self, shape, input_dtype=np.complex64, output_dtype=np.float32):
        """
        Creates a complex-fully-connected dense graph using a shape as parameter
        :param input_dtype: Set to np.float32 to make a real-valued neural network (output_dtype should also be float32)
        :param output_dtype: Datatype of the output of the network. Normally float32 for classification.
            NOTE: If float32 make sure the last activation function gives a float32 and not a complex32!
        :param shape: List of tuple
            1. each number of shape[i][0] correspond to the total neurons of layer i.
            2. a string in shape[i][1] corresponds to the activation function listed on
                https://complex-valued-neural-networks.readthedocs.io/en/latest/act_fun.html
            Where i = 0 corresponds to the input layer and the last value of the list corresponds to the output layer.
        :return: None
        """
        if output_dtype == np.complex64 and input_dtype == np.float32:
            sys.exit("Cvnn::create_mlp_graph: if input dtype is real output cannot be complex")
        # Reset latest graph
        tf.compat.v1.reset_default_graph()

        # Creates the feedforward network
        self.y_out, variables = self._create_graph_from_shape(shape, input_dtype, output_dtype)
        # Defines the loss function
        self.loss = self._categorical_crossentropy_loss()  # TODO: make the user to be able to select the loss!!!!
        # Calculate gradients
        with tf.compat.v1.name_scope("gradients") as scope:
            gradients = tf.gradients(ys=self.loss, xs=variables)
        # Defines a training operator for each variable
        self.training_op = []
        with tf.compat.v1.variable_scope("learning_rule") as scope:
            # lr_const = tf.constant(self.learning_rate, name="learning_rate")
            for i, var in enumerate(variables):
                # Only gradient descent supported for the moment
                self.training_op.append(tf.compat.v1.assign(var, var - self.learning_rate * gradients[i]))
        # assert len(self.training_op) == len(gradients)

        # logs to be saved with tensorboard
        # TODO: add more info like weights
        if self.tensorboard:
            self.writer = tf.compat.v1.summary.FileWriter(self.tbdir, tf.compat.v1.get_default_graph())
            loss_summary = tf.compat.v1.summary.scalar(name='loss_summary', tensor=self.loss)
            self.merged = tf.compat.v1.summary.merge_all()

        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()

        # create saver object of the models weights
        self.saver = tf.compat.v1.train.Saver()
        # for i, var in enumerate(self.saver._var_list):
        #     print('Var {}: {}'.format(i, var))

    def create_linear_regression_graph(self, input_size, output_size,
                                       input_dtype=np.complex64, output_dtype=np.float32):
        """
        Creates a linear_regression_graph with no activation function
        :param input_size:
        :param output_size:
        :param input_dtype:
        :param output_dtype:
        :return:
        """
        self.create_mlp_graph([(input_size, act_linear), (output_size, act_linear)], input_dtype, output_dtype)

    """
    # Loss functions
    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    """
    def _mean_square_loss(self):
        """
        Mean Squared Error, or L2 loss.
        :return:
        """
        with tf.compat.v1.name_scope("loss") as scope:
            error = self.y - self.y_out
            return tf.reduce_mean(input_tensor=tf.square(tf.abs(error)), name="loss")

    def _categorical_crossentropy_loss(self):
        """
        https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        :return: -y*log(y_out)-(1-y)*log(1-y_out) where:
            log - the natural log
            y - binary indicator (0 or 1), it will be all 0's but one (according to the corresponding class)
            y_out - predicted probability observation the class
        """
        with tf.compat.v1.name_scope("loss_scope") as scope:
            y1_error = tf.math.multiply(-self.y, tf.math.log(self.y_out))       # Error for y = 1
            y0_error = tf.math.multiply(1-self.y, tf.math.log(1-self.y_out))    # Error for y = 0
            error = tf.math.subtract(y1_error, y0_error)
            return tf.reduce_mean(input_tensor=error, name="loss")

    # Others
    def restore_graph_from_meta(self, latest_file=None):
        """
        Restores an existing graph from meta data file
        :param latest_file: Path to the file to be restored. If no latest_file given and self.automatic_restore is True,
                            the function will try to load the newest metadata inside `saved_models/` folder.
        :return: None
        """
        if latest_file is None and self.automatic_restore:  # Get the metadata file
            if os.listdir(self.root_savedir):
                print("Getting last model")
                # get newest folder
                list_of_folders = glob.glob(self.root_savedir + '/*')
                latest_folder = max(list_of_folders, key=os.path.getctime)
                # get newest file in the newest folder
                list_of_files = glob.glob(latest_folder + '/*.ckpt.meta')  # Just take ckpt files, not others.
                latest_file = max(list_of_files, key=os.path.getctime)     # .replace('/', '\\')
            else:
                print('Warning:restore_graph_from_meta(): No model found...')
                return None
        elif latest_file is None:
            sys.exit("Error:restore_graph_from_meta(): no latest_file given and automatic_restore disabled")
        # TODO: check latest_file exists and has the correct format!

        # delete the current graph
        tf.compat.v1.reset_default_graph()

        # import the graph from the file
        imported_graph = tf.compat.v1.train.import_meta_graph(latest_file)
        self.restored_meta = True

        # list all the tensors in the graph
        if DEBUG_RESTORE_META:
            for tensor in tf.compat.v1.get_default_graph().get_operations():
                print(tensor.name)

        self.sess = tf.compat.v1.Session()
        with self.sess.as_default():
            imported_graph.restore(self.sess, latest_file.split('.ckpt')[0]+'.ckpt')
            graph = tf.compat.v1.get_default_graph()
            self.loss = graph.get_operation_by_name("loss/loss").outputs[0]
            self.X = graph.get_tensor_by_name("X:0")
            self.y = graph.get_tensor_by_name("Y:0")
            self.y_out = graph.get_tensor_by_name("forward_phase/y_out:0")
            # print(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "learning_rule"))
            self.training_op = [graph.get_operation_by_name(tensor.name) for tensor in
                                tf.compat.v1.get_default_graph().get_operations()
                                if tensor.name.startswith("learning_rule/AssignVariableOp")]
            # logs
            if self.tensorboard:
                self.writer = tf.compat.v1.summary.FileWriter(self.tbdir, tf.compat.v1.get_default_graph())
                self.loss_summary = tf.compat.v1.summary.scalar(name='loss_summary', tensor=self.loss)
                self.merged = tf.compat.v1.summary.merge_all()

            # create saver object
            self.saver = tf.compat.v1.train.Saver()
            # for i, var in enumerate(self.saver._var_list):
            #     print('Var {}: {}'.format(i, var))

    def _init_weights(self, latest_file=None):
        """
        Check for any saved weights within the "saved_models" folder.
        If no model available it initialized the weighs itself.
        If the graph was already restored then the weights are already initialized so the function does nothing.
        :return: None
        """
        if not self.restored_meta:
            with self.sess.as_default():
                assert tf.compat.v1.get_default_session() is self.sess
                if latest_file is None and self.automatic_restore:
                    if os.listdir(self.root_savedir):
                        if self.verbose:
                            print("Cvnn::init_weights: Getting last model")
                        # get newest folder
                        list_of_folders = glob.glob(self.root_savedir + '/*')
                        latest_folder = max(list_of_folders, key=os.path.getctime)
                        # get newest file in the newest folder
                        list_of_files = glob.glob(latest_folder + '/*.ckpt.data*')  # Just take ckpt files, not others.
                        # latest_file = max(list_of_files, key=os.path.getctime).replace('/', '\\')
                        # .split('.ckpt')[0] + '.ckpt'
                        latest_file = max(list_of_files, key=os.path.getctime).split('.ckpt')[0] + '.ckpt'
                        if DEBUG_WEIGHT_LOADER:
                            print("Restoring model: " + latest_file)
                        self.saver.restore(self.sess, latest_file)
                    else:
                        if self.verbose:
                            print("Cvnn::init_weights: No model found.", end='')
                # Check again to see if I found one
                if latest_file is not None:    # TODO: check file exists and has correct format!
                    if DEBUG_WEIGHT_LOADER:
                        print("Restoring model: " + latest_file)
                    self.saver.restore(self.sess, latest_file)
                else:
                    if self.verbose:
                        print("Initializing weights...")
                    self.sess.run(self.init)

    """-----------------
    # Checkpoint methods
    -----------------"""
    def run_checkpoint(self, epoch, num_tr_iter, iteration, feed_dict_batch):
        """
        Calculate and display the batch loss and accuracy. Saves data to tensorboard and saves state of the network
        :param epoch:
        :param num_tr_iter:
        :param iteration:
        :param feed_dict_batch:
        :return:
        """
        loss_batch = self.sess.run(self.loss, feed_dict=feed_dict_batch)
        if self.verbose:
            print("epoch {0:3d}:\t iteration {1:3d}:\t Loss={2:.2f}".format(epoch, iteration, loss_batch))
        # save the model
        self.save_model(epoch, iteration, loss_batch)
        self.save_to_tensorboard(epoch, num_tr_iter, iteration, feed_dict_batch)

    def save_to_tensorboard(self, epoch, num_tr_iter, iteration, feed_dict_batch):
        with self.sess.as_default():
            assert tf.compat.v1.get_default_session() is self.sess
            if self.tensorboard:
                # add the summary to the writer (i.e. to the event file)
                step = epoch * num_tr_iter + iteration
                # if step % num_tr_iter == 0:   # TODO: this must be a function of the display frequency
                #   # Under this case I can plot the x axis as the epoch for clarity
                #    step = epoch
                summary = self.sess.run(self.merged, feed_dict=feed_dict_batch)
                self.writer.add_summary(summary, step)

    def save_model(self, epoch, iteration, loss_batch):
        modeldir = "{}epoch{}-iteration{}-loss{}.ckpt".format(self.savedir, epoch, iteration,
                                                              str(loss_batch).replace('.', ','))
        saved_path = self.saver.save(self.sess, modeldir)
        # print('model saved in {}'.format(saved_path))

    def print_validation_loss(self, epoch, x, y):
        feed_dict_valid = {self.X: x, self.y: y}
        loss_valid = self.sess.run(self.loss, feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.4f}".format(epoch, loss_valid))
        print('---------------------------------------------------------')

    """-------------------
    # Activation functions
    -------------------"""
    @staticmethod
    def apply_activation(act, out):
        """
        Applies activation function `act` to variable `out`
        :param out: Tensor to whom the activation function will be applied
        :param act: function to be applied to out. See the list fo possible activation functions on:
            https://complex-valued-neural-networks.readthedocs.io/en/latest/act_fun.html
        :return: Tensor with the applied activation function
        """
        if callable(act):
            return act(out)         # TODO: for the moment is not be possible to give parameters like alpha
        else:
            print("WARNING: Cvnn::apply_function: " + str(act) + " is not callable, ignoring it")
            return out

    """-----------
    # Initializers
    # https://keras.io/initializers/
    -----------"""
    @staticmethod
    def glorot_uniform_init(in_neurons, out_neurons):
        return np.random.randn(in_neurons, out_neurons) * np.sqrt(1/in_neurons)

    @staticmethod
    def rand_init_neg(in_neurons, out_neurons):
        return 2*np.random.rand(in_neurons, out_neurons)-1

    @staticmethod
    def rand_init(in_neurons, out_neurons):
        """
        Use this function to make fashion not to predict good
        :param in_neurons:
        :param out_neurons:
        :return:
        """
        return np.random.rand(in_neurons, out_neurons)

    """------------
    # Data Analysis
     -----------"""
    # TODO


if __name__ == "__main__":
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 100000
    n = 1000
    num_classes = 4
    x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n, num_classes)

    # Network Declaration
    auto_restore = False
    cvnn = Cvnn("CVNN_tensorboard_debug", automatic_restore=auto_restore)

    input_size = np.shape(x_train)[1]
    hidden_size = 10
    output_size = np.shape(y_train)[1]
    if not auto_restore:
        # cvnn.create_linear_regression_graph(input_size, output_size)
        cvnn.create_mlp_graph([(input_size, act_linear),
                               (hidden_size, act_cart_sigmoid),
                               (output_size, act_cart_softmax_real)])

    cvnn.train(x_train, y_train, x_test, y_test)

    """y_out = cvnn.predict(x_test)
    if y_out is not None:
        print(y_out[:3])
        print(y_test[:3])"""

