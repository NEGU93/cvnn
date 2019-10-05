import tensorflow as tf
import data_processing as dp
from datetime import datetime
import numpy as np
import glob
import os

DEBUGGER = False


class Cvnn:
    def __init__(self, input_size=20, output_size=2, learning_rate=0.001, tensorboard=True):
        tf.compat.v1.disable_eager_execution()

        self.input_size = input_size
        self.output_size = output_size

        # Hyper-parameters
        self.epochs = 100                       # Total number of training epochs
        self.batch_size = 100                   # Training batch size
        self.display_freq = 1000                # Display results frequency
        self.learning_rate = learning_rate      # The optimization initial learning rate

        # logs dir
        self.tensorboard = tensorboard
        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = "log"
        self.logdir = "{}/run-{}/".format(self.root_logdir, self.now)

        # Use date for creating different files at each run.
        self.root_savedir = "saved_models"
        self.savedir = "{}/run-{}/".format(self.root_savedir, self.now)
        if not os.path.exists(self.root_savedir):
            os.makedirs(self.root_savedir)

        self.create_linear_regression_graph()
        # self.create_2_layer_graph()

        # create saver object
        # TODO: here it's ok?
        self.saver = tf.compat.v1.train.Saver()
        # for i, var in enumerate(self.saver._var_list):
        #     print('Var {}: {}'.format(i, var))

    def create_linear_regression_graph(self):
        # Reset latest graph
        tf.compat.v1.reset_default_graph()

        # Define placeholders
        self.X = tf.compat.v1.placeholder(tf.complex64, shape=[None, input_size], name='X')
        self.y = tf.compat.v1.placeholder(tf.complex64, shape=[None, output_size], name='Y')

        # Create weight matrix initialized randomely from N~(0, 0.01)
        self.w = tf.Variable(tf.complex(np.random.rand(input_size, output_size).astype(np.float32),
                                   np.random.rand(input_size, output_size).astype(np.float32)), name="weights")
        self.b = tf.Variable(tf.complex(np.random.rand(output_size).astype(np.float32),
                                   np.random.rand(output_size).astype(np.float32)), name="bias")

        with tf.compat.v1.name_scope("forward_phase") as scope:
            self.y_out = tf.add(tf.matmul(self.X, self.w), self.b)

        # Define Graph
        with tf.compat.v1.name_scope("loss") as scope:
            self.error = self.y - self.y_out
            self.loss = tf.reduce_mean(input_tensor=tf.square(tf.abs(self.error)), name="mse")

        with tf.compat.v1.name_scope("leargning_rule") as scope:
            self.gradients_w, self.gradients_b = tf.gradients(ys=self.loss, xs=[self.w, self.b])
            self.training_op_w = tf.compat.v1.assign(self.w, self.w - self.learning_rate * self.gradients_w)
            self.training_op_b = tf.compat.v1.assign(self.b, self.b - self.learning_rate * self.gradients_b)
            self.training_op = [self.training_op_w, self.training_op_b]

        # logs
        if self.tensorboard:
            self.writer = tf.compat.v1.summary.FileWriter(self.logdir, tf.compat.v1.get_default_graph())
            self.loss_summary = tf.compat.v1.summary.scalar(name='loss_summary', tensor=self.loss)
            self.real_weight_summary = tf.compat.v1.summary.histogram('real_weight_summary',
                                                             tf.math.real(self.w))  # cannot pass complex
            self.imag_weight_summary = tf.compat.v1.summary.histogram('imag_weight_summary', tf.math.imag(self.w))
            self.merged = tf.compat.v1.summary.merge_all()

        self.init = tf.compat.v1.global_variables_initializer()

    def train(self, x_train, y_train, x_test, y_test):
        with tf.compat.v1.Session() as sess:
            if os.listdir(self.root_savedir):
                print("Getting last model")
                # get newest folder
                list_of_folders = glob.glob(self.root_savedir + '/*')
                latest_folder = max(list_of_folders, key=os.path.getctime)
                # get newest file in the newest folder
                list_of_files = glob.glob(latest_folder + '/*.ckpt.data*')  # Just take ckpt files, not others.
                # latest_file = max(list_of_files, key=os.path.getctime).replace('/', '\\').split('.ckpt')[0] + '.ckpt'
                latest_file = max(list_of_files, key=os.path.getctime).split('.ckpt')[0] + '.ckpt'
                # import pdb; pdb.set_trace()
                self.saver.restore(sess, latest_file)
            else:
                print("No model found, initializing weights")
                sess.run(self.init)

            # Number of training iterations in each epoch
            num_tr_iter = int(len(y_train) / self.batch_size)
            for epoch in range(self.epochs):
                # Randomly shuffle the training data at the beginning of each epoch
                x_train, y_train = dp.randomize(x_train, y_train)
                for iteration in range(num_tr_iter):
                    start = iteration * self.batch_size
                    end = (iteration + 1) * self.batch_size
                    x_batch, y_batch = dp.get_next_batch(x_train, y_train, start, end)
                    # Run optimization op (backprop)
                    feed_dict_batch = {self.X: x_batch, self.y: y_batch}
                    sess.run(self.training_op, feed_dict=feed_dict_batch)
                    if (epoch * self.batch_size + iteration) % self.display_freq == 0:
                        # Calculate and display the batch loss and accuracy
                        loss_batch = sess.run(self.loss, feed_dict=feed_dict_batch)
                        print("epoch {0:3d}:\t iteration {1:3d}:\t Loss={2:.2f}".format(epoch, iteration, loss_batch))
                        # save the model
                        modeldir = "{}epoch{}-iteration{}-loss{}.ckpt".format(self.savedir, epoch, iteration,
                                                                         str(loss_batch).replace('.', ','))
                        saved_path = self.saver.save(sess, modeldir)
                        # print('model saved in {}'.format(saved_path))
                        if self.tensorboard:
                            # add the summary to the writer (i.e. to the event file)
                            step = epoch * num_tr_iter + iteration
                            if step % num_tr_iter == 0:
                                # Under this case I can plot the x axis as the epoch for clarity
                                step = epoch
                            summary = sess.run(self.merged, feed_dict=feed_dict_batch)
                            self.writer.add_summary(summary, step)

            # Run validation after every epoch
            feed_dict_valid = {self.X: x_test, self.y: y_test}
            loss_valid = sess.run(self.loss, feed_dict=feed_dict_valid)
            print('---------------------------------------------------------')
            print("Epoch: {0}, validation loss: {1:.4f}".format(epoch + 1, loss_valid))
            print('---------------------------------------------------------')
            if DEBUGGER:
                print(y_test[:3])
                print(self.y_out.eval(feed_dict=feed_dict_valid)[:3])
        if self.tensorboard:
            self.writer.close()

    def predict(self, x):
        with tf.compat.v1.Session() as sess:
            if os.listdir(self.root_savedir):
                print("Getting last model")
                # get newest folder
                list_of_folders = glob.glob(self.root_savedir + '/*')
                latest_folder = max(list_of_folders, key=os.path.getctime)
                # get newest file in the newest folder
                list_of_files = glob.glob(latest_folder + '/*.ckpt.data*')  # Just take ckpt files, not others.
                # latest_file = max(list_of_files, key=os.path.getctime).replace('/', '\\').split('.ckpt')[0] + '.ckpt'
                latest_file = max(list_of_files, key=os.path.getctime).split('.ckpt')[0] + '.ckpt'
                # import pdb; pdb.set_trace()
                self.saver.restore(sess, latest_file)
            else:
                print("No model found, please train your model first")
                return None

            feed_dict_valid = {self.X: x}
            return self.y_out.eval(feed_dict=feed_dict_valid)


if __name__ == "__main__":
    # Data pre-processing
    m = 5000
    n = 30
    input_size = n
    output_size = 1
    total_cases = 2*m
    train_ratio = 0.8
    # x_train, y_train, x_test, y_test = dp.get_non_correlated_gaussian_noise(m, n)

    x_input = np.random.rand(total_cases, input_size) + 1j * np.random.rand(total_cases, input_size)
    w_real = np.random.rand(input_size, output_size) + 1j * np.random.rand(input_size, output_size)
    desired_output = np.matmul(x_input, w_real)  # Generate my desired output

    # Separate train and test set
    x_train = x_input[:int(train_ratio * total_cases), :]
    y_train = desired_output[:int(train_ratio * total_cases), :]
    x_test = x_input[int(train_ratio * total_cases):, :]
    y_test = desired_output[int(train_ratio * total_cases):, :]

    # Network Declaration
    cvnn = Cvnn(input_size=n, output_size=output_size)

    cvnn.train(x_train, y_train, x_test, y_test)
    y_out = cvnn.predict(x_test)
    if y_out is not None:
        print(y_out[:3])
        print(y_test[:3])

