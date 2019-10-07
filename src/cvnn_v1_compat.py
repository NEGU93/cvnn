import tensorflow as tf
import data_processing as dp
from datetime import datetime
import numpy as np
import glob
import sys
import os

DEBUGGER = False
DEBUG_SAVER = False
DEBUG_RETORE_META = False


class Cvnn:
    # Constructor and Destructor
    def __init__(self, learning_rate=0.001,
                 tensorboard=True, verbose=True, automatic_restore=True):
        tf.compat.v1.disable_eager_execution()

        self.verbose = verbose
        self.automatic_restore = automatic_restore
        self.tensorboard = tensorboard

        # Hyper-parameters
        self.learning_rate = learning_rate      # The optimization initial learning rate

        # logs dir
        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = "../log"
        self.logdir = "{}/run-{}/".format(self.root_logdir, self.now)

        # Use date for creating different files at each run.
        self.root_savedir = "../saved_models"
        self.savedir = "{}/run-{}/".format(self.root_savedir, self.now)
        if not os.path.exists(self.root_savedir):
            os.makedirs(self.root_savedir)

        # self.create_linear_regression_graph()
        # Launch the graph in a session.
        # self.sess = tf.compat.v1.Session()
        self.restored_meta = False
        if automatic_restore:
            self.restore_graph_from_meta()

    def __del__(self):
        if self.tensorboard:
            try:
                self.writer.close()
            except:  # Get the real exception.
                print("Writer did not exist, couldn't delete it")
        self.sess.close()

    # Train and predict models
    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=100, display_freq=1000):
        """
        Performs the training of the neural network
        :param x_train: Training data
        :param y_train: Labels of the training data
        :param x_test: Test data to display accuracy at the end
        :param y_test: Test labels
        :param epochs: Total number of training epochs
        :param batch_size: Training batch size
        :param display_freq: Display results frequency
        :return: None
        """
        with self.sess.as_default():
            assert tf.compat.v1.get_default_session() is self.sess
            self.init_weights()

            # Run validation at beginning
            self.print_validation_loss(0, x_test, y_test)
            # Number of training iterations in each epoch
            num_tr_iter = int(len(y_train) / batch_size)
            for epoch in range(epochs):
                # Randomly shuffle the training data at the beginning of each epoch
                x_train, y_train = dp.randomize(x_train, y_train)
                for iteration in range(num_tr_iter):
                    start = iteration * batch_size
                    end = (iteration + 1) * batch_size
                    x_batch, y_batch = dp.get_next_batch(x_train, y_train, start, end)
                    # Run optimization op (backprop)
                    feed_dict_batch = {self.X: x_batch, self.y: y_batch}
                    self.sess.run(self.training_op, feed_dict=feed_dict_batch)
                    if (epoch * batch_size + iteration) % display_freq == 0:
                        self.run_checkpoint(epoch, num_tr_iter, iteration, feed_dict_batch)

            # Run validation at the end
            feed_dict_valid = {self.X: x_test, self.y: y_test}
            loss_valid = self.sess.run(self.loss, feed_dict=feed_dict_valid)
            self.print_validation_loss(epoch+1, x_test, y_test)
            self.save_model("final", "valid_loss", loss_valid)
            if DEBUGGER:
                print(y_test[:3])
                print(self.y_out.eval(feed_dict=feed_dict_valid)[:3])

    def predict(self, x):
        """
        Runs a single feedforward computation
        :param x: Input of the network
        :return: Output of the network
        """
        with self.sess.as_default():
            assert tf.compat.v1.get_default_session() is self.sess
            feed_dict_valid = {self.X: x}
            return self.y_out.eval(feed_dict=feed_dict_valid)

    # Graph creation
    def restore_graph_from_meta(self, latest_file=None):
        # Get the metadata file
        if latest_file is None and self.automatic_restore:
            if os.listdir(self.root_savedir):
                print("Getting last model")
                # get newest folder
                list_of_folders = glob.glob(self.root_savedir + '/*')
                latest_folder = max(list_of_folders, key=os.path.getctime)
                # get newest file in the newest folder
                list_of_files = glob.glob(latest_folder + '/*.ckpt.meta')  # Just take ckpt files, not others.
                latest_file = max(list_of_files, key=os.path.getctime)     # .replace('/', '\\')
            else:
                sys.exit('Error:restore_graph_from_meta(): No model found...')
        else:
            sys.exit("Error:restore_graph_from_meta(): no latest_file given and automatic_restore disabled")
        # TODO: check latest_file exists and has the correct format!

        # delete the current graph
        tf.compat.v1.reset_default_graph()

        # import the graph from the file
        imported_graph = tf.compat.v1.train.import_meta_graph(latest_file)
        self.restored_meta = True

        # list all the tensors in the graph
        if DEBUG_RETORE_META:
            for tensor in tf.compat.v1.get_default_graph().get_operations():
                print(tensor.name)

        self.sess = tf.compat.v1.Session()
        with self.sess.as_default():
            imported_graph.restore(self.sess, latest_file.split('.ckpt')[0]+'.ckpt')
            graph = tf.compat.v1.get_default_graph()
            self.loss = graph.get_operation_by_name("loss/loss").outputs[0]
            self.X = graph.get_tensor_by_name("X:0")
            self.y = graph.get_tensor_by_name("Y:0")
            self.w = graph.get_tensor_by_name("weights:0")
            self.b = graph.get_tensor_by_name("bias:0")
            self.y_out = graph.get_tensor_by_name("forward_phase/y_out:0")
            # print(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "learning_rule"))
            self.training_op = [graph.get_operation_by_name(tensor.name) for tensor in
                                tf.compat.v1.get_default_graph().get_operations()
                                if tensor.name.startswith("learning_rule/AssignVariableOp")]
            # self.training_op = [graph.get_operation_by_name("leargning_rule/AssignVariableOp"),
                                #graph.get_operation_by_name("leargning_rule/AssignVariableOp_1")]
            # logs
            if self.tensorboard:
                self.writer = tf.compat.v1.summary.FileWriter(self.logdir, tf.compat.v1.get_default_graph())
                self.loss_summary = tf.compat.v1.summary.scalar(name='loss_summary', tensor=self.loss)
                self.merged = tf.compat.v1.summary.merge_all()

            # create saver object
            self.saver = tf.compat.v1.train.Saver()
            # for i, var in enumerate(self.saver._var_list):
            #     print('Var {}: {}'.format(i, var))

    def create_complex_dense(self, input_size, output_size, input):
        w = tf.Variable(tf.complex(np.random.rand(input_size, output_size).astype(np.float32),
                                        np.random.rand(input_size, output_size).astype(np.float32)), name="weights")
        b = tf.Variable(tf.complex(np.random.rand(output_size).astype(np.float32),
                                        np.random.rand(output_size).astype(np.float32)), name="bias")

        return tf.add(tf.matmul(input, w), b), [w, b]

    def create_linear_regression_graph(self, input_size, output_size):
        # Reset latest graph
        tf.compat.v1.reset_default_graph()

        # Define placeholders
        self.X = tf.compat.v1.placeholder(tf.complex64, shape=[None, input_size], name='X')
        self.y = tf.compat.v1.placeholder(tf.complex64, shape=[None, output_size], name='Y')

        # Create weight matrix initialized randomely from N~(0, 0.01)
        self.y_out, variables = self.create_complex_dense(input_size, output_size, self.X)

        # with tf.compat.v1.name_scope("forward_phase") as scope:
        print(variables)
        # Define Graph
        with tf.compat.v1.name_scope("loss") as scope:
            self.error = self.y - self.y_out
            self.loss = tf.reduce_mean(input_tensor=tf.square(tf.abs(self.error)), name="loss")
        with tf.compat.v1.name_scope("gradients") as scope:
            self.gradients = tf.gradients(ys=self.loss, xs=variables)
        self.training_op = []
        with tf.compat.v1.name_scope("learning_rule") as scope:
            for i, var in enumerate(variables):
                self.training_op.append(tf.compat.v1.assign(var, var - self.learning_rate * self.gradients[i]))
        # print(self.training_op)

        # logs
        if self.tensorboard:
            self.writer = tf.compat.v1.summary.FileWriter(self.logdir, tf.compat.v1.get_default_graph())
            self.loss_summary = tf.compat.v1.summary.scalar(name='loss_summary', tensor=self.loss)
            # self.real_weight_summary = tf.compat.v1.summary.histogram('real_weight_summary',
                                  #                           tf.math.real(self.w))  # cannot pass complex
            # self.imag_weight_summary = tf.compat.v1.summary.histogram('imag_weight_summary', tf.math.imag(self.w))
            self.merged = tf.compat.v1.summary.merge_all()
            # print(self.merged)

        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        # create saver object
        self.saver = tf.compat.v1.train.Saver()
        # for i, var in enumerate(self.saver._var_list):
        #     print('Var {}: {}'.format(i, var))

    def init_weights(self, latest_file=None):
        """
        Check for any saved weights within the "saved_models" folder.
        If no model available it initialized the weighs itself.
        If the graph was already restored then the weights are already initialized so the function does nothing.
        :return: None
        """
        if not self.restored_meta:
            with self.sess.as_default():
                assert tf.compat.v1.get_default_session() is self.sess
                if latest_file is None: # and self.automatic_restore:
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
                        if DEBUG_SAVER:
                            print("Restoring model: " + latest_file)
                        self.saver.restore(self.sess, latest_file)
                # Check again to see if I found one
                if latest_file is not None:    # TODO: check file exists and has correct format!
                    if DEBUG_SAVER:
                        print("Restoring model: " + latest_file)
                    self.saver.restore(self.sess, latest_file)
                else:
                    if self.verbose:
                        print("Cvnn::init_weights: No model found, initializing weights")
                    self.sess.run(self.init)

    # Checkpoint methods
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
                if step % num_tr_iter == 0:
                    # Under this case I can plot the x axis as the epoch for clarity
                    step = epoch
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

    # Activation functions
    def act_null(self, z):
        return z


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = dp.load_dataset("linear_output")

    input_size = np.shape(x_train)[1]
    output_size = np.shape(y_train)[1]

    # Network Declaration
    cvnn = Cvnn(automatic_restore=False)

    cvnn.create_linear_regression_graph(input_size, output_size)

    cvnn.train(x_train, y_train, x_test, y_test)
    y_out = cvnn.predict(x_test)
    if y_out is not None:
        print(y_out[:3])
        print(y_test[:3])

