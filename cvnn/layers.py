from abc import ABC, abstractmethod
from itertools import count
import tensorflow as tf
import sys
import cvnn
import logging
from typing import Union, Optional          # https://docs.python.org/3/library/typing.html
from cvnn.activation_functions import apply_activation
from cvnn.utils import get_func_name
import numpy as np
from tensorflow.keras import layers
from pdb import set_trace

# Initializers:
# https://www.tensorflow.org/api_docs/python/tf/keras/initializers
# https://keras.io/initializers/

SUPPORTED_DTYPES = (np.complex64, np.float32)  # , np.complex128, np.float64) Gradients return None when complex128
layer_count = count(0)  # Used to count the number of layers

t_Dtype = Union[tf.dtypes.DType, np.dtype]
t_Shape = Union[int, tuple, list]


class ComplexLayer(layers.Layer, ABC):
    # Being ComplexLayer an abstract class, then this can be called using:
    #   self.__class__.__bases__.<variable>
    # As all child's will have this class as base, mro gives a full list so won't work.
    last_layer_output_dtype = None      # TODO: Make it work both with np and tf dtypes
    last_layer_output_size = None

    def __init__(self, output_size: t_Shape, input_size=Optional[t_Shape], input_dtype=t_Dtype):
        """
        Base constructor for a complex layer. The first layer will need a input_dtype and input_size.
        For the other classes is optional,
            if input_size or input_dtype does not match last layer it will throw a warning
        :param output_size: Output size of the layer
        :param input_size: Input size of the layer
        :param input_dtype: data type of the input
        """
        self.logger = logging.getLogger(cvnn.__name__)

        if input_dtype is None and self.__class__.__bases__[0].last_layer_output_dtype is None:
            # None input dtype given but it's the first layer declared
            self.logger.error("First layer must be given an input dtype", exc_info=True)
            sys.exit(-1)
        elif input_dtype is None and self.__class__.__bases__[0].last_layer_output_dtype is not None:
            # Use automatic mode
            self.input_dtype = self.__class__.__bases__[0].last_layer_output_dtype
        elif input_dtype is not None:
            if input_dtype not in SUPPORTED_DTYPES:
                self.logger.error("Layer::__init__: unsupported input_dtype " + str(input_dtype), exc_info=True)
                sys.exit(-1)
            if self.__class__.__bases__[0].last_layer_output_dtype is not None:
                if self.__class__.__bases__[0].last_layer_output_dtype != input_dtype:
                    self.logger.warning("Input dtype " + str(input_dtype) +
                                        " is not equal to last layer's input dtype " +
                                        str(self.__class__.__bases__[0].last_layer_output_dtype))
            self.input_dtype = input_dtype

        # This will be normally the case.
        # Each layer must change this value if needed.
        self.__class__.__bases__[0].last_layer_output_dtype = self.input_dtype

        # Input Size
        if input_size is None:
            if self.__class__.__bases__[0].last_layer_output_size is None:
                # None input size given but it's the first layer declared
                self.logger.error("First layer must be given an input size")
                sys.exit(-1)
            else:  # self.__class__.__bases__[0].last_layer_output_dtype is not None:
                self.input_size = self.__class__.__bases__[0].last_layer_output_size
        elif input_size is not None:
            if self.__class__.__bases__[0].last_layer_output_size is not None:
                if input_size != self.__class__.__bases__[0].last_layer_output_size:
                    self.logger.warning("Input size " + str(input_size) + " is not equal to last layer's output size " +
                                        str(self.__class__.__bases__[0].last_layer_output_size))
            self.input_size = input_size

        self.output_size = output_size
        self.__class__.__bases__[0].last_layer_output_size = self.output_size
        self.layer_number = next(layer_count)  # Know it's own number

        super().__init__()

    def set_output_size(self):
        self.__class__.__bases__[0].last_layer_output_size = self.output_size

    @abstractmethod
    def __deepcopy__(self, memodict=None):
        pass

    def get_input_dtype(self):
        return self.input_dtype

    @abstractmethod
    def get_real_equivalent(self):
        """
        :return: Gets a real-valued COPY of the Complex Layer.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        :return: a string containing all the information of the layer
        """
        pass

    @abstractmethod
    def save_tensorboard_checkpoint(self, x,
                                    weight_summary: tf.summary.SummaryWriter,
                                    activation_summary: tf.summary.SummaryWriter,
                                    step=Optional[int]):
        pass


class ComplexDense(ComplexLayer):
    """
    Fully connected complex-valued layer
    Implements the operation:
        activation(dot(input, weights) + bias)
    - where data types can be either complex or real.
    - activation is the element-wise activation function passed as the activation argument,
    - weights is a matrix created by the layer
    - bias is a bias vector created by the layer
    """

    def __init__(self, output_size, input_size=None, activation=None, input_dtype=None,
                 weight_initializer=tf.keras.initializers.GlorotUniform, bias_initializer=tf.keras.initializers.Zeros,
                 dropout=None):
        """
        Initializer of the Dense layer
        :param output_size: Output size of the layer
        :param input_size: Input size of the layer
        :param activation: Activation function to be used.
            Can be either the function from cvnn.activation or tensorflow.python.keras.activations
            or a string as listed in act_dispatcher
        :param input_dtype: data type of the input. Default: np.complex64
            Supported data types:
                - np.complex64
                - np.float32
        :param weight_initializer: Initializer for the weights.
            Default: tensorflow.keras.initializers.GlorotUniform
        :param bias_initializer: Initializer fot the bias.
            Default: tensorflow.keras.initializers.Zeros
        :param dropout: Either None (default) and no dropout will be applied or a scalar
            that will be the probability that each element is dropped.
            Example: setting rate=0.1 would drop 10% of input elements.
        """
        super(ComplexDense, self).__init__(output_size=output_size, input_size=input_size, input_dtype=input_dtype)
        self.activation = activation
        # Test if the activation function changes datatype or not...
        self.__class__.__bases__[0].last_layer_output_dtype = \
            apply_activation(self.activation,
                             tf.cast(tf.complex([[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]), self.input_dtype)
                             ).numpy().dtype
        self.dropout = dropout
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer  # TODO: Not working yet
        self.w = None
        self.b = None
        self._init_weights()

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return ComplexDense(output_size=self.output_size, input_size=self.input_size,
                            activation=self.activation,
                            input_dtype=self.input_dtype,
                            weight_initializer=self.weight_initializer,
                            bias_initializer=self.bias_initializer, dropout=self.dropout
                            )

    def get_real_equivalent(self, output_multiplier=2):
        """
        :param output_multiplier: Multiplier of output and input size (normally by 2)
        :return: real-valued copy of self
        """
        return ComplexDense(output_size=self.output_size * output_multiplier, input_size=self.input_size * 2,
                            activation=self.activation, input_dtype=np.float32,
                            weight_initializer=self.weight_initializer,
                            bias_initializer=self.bias_initializer, dropout=self.dropout
                            )

    def _init_weights(self):
        if self.input_dtype == np.complex64 or self.input_dtype == np.complex128:  # Complex layer
            self.w = tf.cast(
                tf.Variable(tf.complex(self.weight_initializer()(shape=(self.input_size, self.output_size)),
                                       self.weight_initializer()(shape=(self.input_size, self.output_size))),
                            name="weights" + str(self.layer_number)),
                dtype=self.input_dtype)
            self.b = tf.cast(tf.Variable(tf.complex(self.bias_initializer()(self.output_size),
                                                    self.bias_initializer()(self.output_size)),
                                         name="bias" + str(self.layer_number))
                             , dtype=self.input_dtype)
        elif self.input_dtype == np.float32 or self.input_dtype == np.float64:  # Real Layer
            self.w = tf.cast(tf.Variable(self.weight_initializer()(shape=(self.input_size, self.output_size)),
                                         name="weights" + str(self.layer_number)),
                             dtype=self.input_dtype)
            self.b = tf.cast(tf.Variable(self.bias_initializer()(self.output_size),
                                         name="bias" + str(self.layer_number)),
                             dtype=self.input_dtype)
        else:
            # This case should never happen. The abstract constructor should already have checked this
            self.logger.error("Input_dtype not supported.", exc_info=True)
            sys.exit(-1)

    def get_description(self):
        fun_name = get_func_name(self.activation)
        out_str = "Dense layer:\n\tinput size = " + str(self.input_size) + "(" + str(self.input_dtype) + \
                  ") -> output size = " + str(self.output_size) + \
                  ";\n\tact_fun = " + fun_name + ";\n\tweight init = " \
                  + self.weight_initializer.__name__ + "; bias init = " + self.bias_initializer.__name__ + \
                  "\n\tDropout: " + str(self.dropout) + "\n"
        return out_str

    def call(self, inputs, **kwargs):
        """
        Applies the layer to an input
        :param inputs: input
        :param kwargs:
        :return: result of applying the layer to the inputs
        """
        # TODO: treat bias as a weight. It might optimize training (no add operation, only mult)
        with tf.name_scope("ComplexDense_" + str(self.layer_number)) as scope:
            if tf.dtypes.as_dtype(inputs.dtype) is not tf.dtypes.as_dtype(np.dtype(self.input_dtype)):
                self.logger.warning("Dense::apply_layer: Input dtype " + str(inputs.dtype) + " is not as expected ("
                                    + str(tf.dtypes.as_dtype(np.dtype(self.input_dtype))) +
                                    "). Casting input but you most likely have a bug")
            out = tf.add(tf.matmul(tf.cast(inputs, self.input_dtype), self.w), self.b)
            y_out = apply_activation(self.activation, out)

            if self.dropout:
                # $ tf.nn.dropout(tf.complex(x,x), rate=0.5)
                # *** ValueError: x has to be a floating point tensor since it's going to be scaled.
                # Got a <dtype: 'complex64'> tensor instead.
                drop_filter = tf.nn.dropout(tf.ones(y_out.shape), rate=self.dropout)
                y_out_real = tf.multiply(drop_filter, tf.math.real(y_out))
                y_out_imag = tf.multiply(drop_filter, tf.math.imag(y_out))
                y_out = tf.cast(tf.complex(y_out_real, y_out_imag), dtype=y_out.dtype)
            return y_out

    def save_tensorboard_checkpoint(self, x, weight_summary, activation_summary, step=None):
        self._save_tensorboard_weight(weight_summary, step)
        return self._save_tensorboard_activations(x, activation_summary, step)

    def _save_tensorboard_weight(self, summary, step):
        with summary.as_default():
            if self.input_dtype == np.complex64 or self.input_dtype == np.complex128:
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_w_real",
                                     data=tf.math.real(self.w), step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_w_imag",
                                     data=tf.math.imag(self.w), step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_b_real",
                                     data=tf.math.real(self.b), step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_b_imag",
                                     data=tf.math.imag(self.b), step=step)
            elif self.input_dtype == np.float32 or self.input_dtype == np.float64:
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_w",
                                     data=self.w, step=step)
                tf.summary.histogram(name="ComplexDense_" + str(self.layer_number) + "_b",
                                     data=self.b, step=step)
            else:
                # This case should never happen. The constructor should already have checked this
                self.logger.error("Input_dtype not supported.", exc_info=True)
                sys.exit(-1)

    def _save_tensorboard_activations(self, x, summary, step):
        x = self.call(x)
        with summary.as_default():
            if x.dtype == tf.complex64 or x.dtype == tf.complex128:
                tf.summary.histogram(name="Activation_value_" + str(self.layer_number) + "_real",
                                     data=tf.math.real(x), step=step)
                tf.summary.histogram(name="Activation_value_" + str(self.layer_number) + "_imag",
                                     data=tf.math.imag(x), step=step)
            elif x.dtype == tf.float32 or x.dtype == tf.float64:
                tf.summary.histogram(name="Activation_value_" + str(self.layer_number),
                                     data=x, step=step)
            else:
                self.logger.error("Input_dtype not supported. Should never have gotten here!", exc_info=True)
                sys.exit(-1)
        return x


class ComplexDropout(ComplexLayer):

    def __init__(self, rate, noise_shape=None, seed=None):
        """
        :param rate: A scalar Tensor with the same type as x.
            The probability that each element is dropped.
            For example, setting rate=0.1 would drop 10% of input elements.
        :param noise_shape: A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
        :param seed:  A Python integer. Used to create random seeds. See tf.random.set_seed for behavior.
        """
        # tf.random.set_seed(seed)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        # Win x2: giving None as input_size will also make sure Dropout is not the first layer
        super().__init__(input_size=None, output_size=None, input_dtype=None)

    def call(self, inputs, **kwargs):
        drop_filter = tf.nn.dropout(tf.ones(inputs.shape), rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)
        y_out_real = tf.multiply(drop_filter, tf.math.real(inputs))
        y_out_imag = tf.multiply(drop_filter, tf.math.imag(inputs))
        return tf.cast(tf.complex(y_out_real, y_out_imag), dtype=inputs.dtype)

    def save_tensorboard_checkpoint(self, x, weight_summary, activation_summary, step=None):
        # No tensorboard things to save
        return None

    def get_description(self):
        return "Complex Dropout:\n\trate={}".format(self.rate)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return ComplexDropout(rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)

    def get_real_equivalent(self):
        return self.__deepcopy__()      # Dropout layer is dtype agnostic


class ComplexConvolutional(ComplexLayer):

    def __init__(self, filters: int, kernel_shape: t_Shape,
                 input_shape: Optional[t_Shape] = None, padding: t_Shape = 0,
                 stride: t_Shape = 1, input_dtype=t_Dtype,
                 activation=None    # TODO: Check datatype
                 ):
        self.filters = filters
        self.activation = activation
        # Test if the activation function changes datatype or not...
        self.__class__.__bases__[0].last_layer_output_dtype = \
            apply_activation(self.activation,
                             tf.cast(tf.complex([[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]), input_dtype)
                             ).numpy().dtype
        if input_shape is None:
            self.input_size = None
        elif isinstance(input_shape, int):
            self.input_size = (input_shape,)
        elif isinstance(input_shape, (tuple, list)):
            self.input_size = tuple(input_shape)
        else:
            self.logger.error(
                "Input shape: " + str(input_shape) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        super(ComplexConvolutional, self).__init__(output_size=None, input_size=self.input_size,
                                                   input_dtype=input_dtype)
        if isinstance(kernel_shape, int):
            self.kernel_shape = (kernel_shape,) * len(self.input_size)
        elif isinstance(kernel_shape, (tuple, list)):
            self.kernel_shape = tuple(kernel_shape)
        else:
            self.logger.error(
                "Kernel shape: " + str(kernel_shape) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        # Padding
        if isinstance(padding, int):
            self.padding_shape = (padding,) * len(self.input_size)   # I call super first in the case input_shape is none
        elif isinstance(padding, (tuple, list)):
            self.padding_shape = tuple(padding)
        else:
            self.logger.error("Padding: " + str(padding) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        # Stride
        if isinstance(stride, int):
            self.stride_shape = (stride,) * len(self.input_size)   # I call super first in the case input_shape is none
        elif isinstance(padding, (tuple, list)):
            self.stride_shape = tuple(stride)
        else:
            self.logger.error("stride: " + str(stride) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        out_list = []
        for i in range(len(self.input_size)):
            # 2.4 on https://arxiv.org/abs/1603.07285
            out_list.append(int(np.floor(
                (self.input_size[i] + 2 * self.padding_shape[i] - self.kernel_shape[i]) / self.stride_shape[i]
            ) + 1))
        self.output_size = tuple(out_list)
        self.set_output_size()  # TODO: Not a nice fix
        self._init_kernel()

    def _init_kernel(self):
        self.kernels = []
        if self.input_dtype == np.complex64 or self.input_dtype == np.complex128:  # Complex layer
            for _ in range(self.filters):
                self.kernels.append(tf.cast(
                    tf.Variable(tf.complex(tf.keras.initializers.GlorotUniform()(shape=self.kernel_shape),
                                           tf.keras.initializers.GlorotUniform()(shape=self.kernel_shape)),
                                name="kernel" + str(self.layer_number)),
                    dtype=self.input_dtype))
        elif self.input_dtype == np.float32 or self.input_dtype == np.float64:  # Real Layer
            for _ in range(self.filters):
                self.kernels.append(tf.cast(tf.Variable(tf.keras.initializers.GlorotUniform()(shape=self.kernel_shape),
                                                        name="kernel" + str(self.layer_number)),
                                            dtype=self.input_dtype))
        else:
            # This case should never happen. The abstract constructor should already have checked this
            self.logger.error("Input_dtype not supported.", exc_info=True)
            sys.exit(-1)

    def _verify_inputs(self, inputs):
        # Expected inputs shape: (images, image_shape, channel (optional))
        inputs = tf.convert_to_tensor(inputs)  # This checks all images are same size! Nice
        if len(inputs.shape) == len(self.input_size) + 1:
            # case with no channel
            inputs = tf.reshape(inputs, inputs.shape + (1,))    # Then I have only one channel, I add dimension
        elif len(inputs.shape) != len(self.input_size) + 2:     # This is the other expected input.
            self.logger.error("inputs.shape should at least be of size 3 (case of 1D inputs) "
                              "with the shape of (images, channels, vector size)")
            sys.exit(-1)
        if inputs.shape[1:-1] != self.input_size:   # Remove # of images (index 0) and remove channels (index -1)
            expected_shape = "(images, " + str(self.input_size[0])
            for i in range(1, len(self.input_size)):
                expected_shape += "x" + str(self.input_size[i])
            expected_shape += ", channels (optional)) "

            received_shape = "(images=" + str(inputs.shape[0]) + ", "
            received_shape += str(inputs.shape[1])  # What if it has different sizes????
            for i in range(2, len(inputs.shape)-1):
                received_shape += "x" + str(inputs.shape[i])
            received_shape += ", channels=" + str(inputs.shape[-1]) + ")"
            self.logger.error("Unexpected image shape. Expecting image of shape " +
                              expected_shape + " but received " + received_shape)
            sys.exit(-1)
        return inputs

    def call(self, inputs, **kwargs):
        with tf.name_scope("ComplexConvolution_" + str(self.layer_number)) as scope:
            inputs = self._verify_inputs(inputs)
            inputs = self.apply_padding(inputs)
            output = np.zeros(
                (inputs.shape[0],) +                        # Per each image
                self.output_size +                          # Image out size
                (self.filters*inputs.shape[-1],)            # New channels
            )
            for img_index, image in enumerate(inputs):
                for channel_index in range(image.shape[-1]):
                    img_channel = image[..., channel_index]     # Get each channel
                    for filter_index, kernel in enumerate(self.kernels):
                        for i in range(int(np.prod(self.output_size))):  # for each element in the output
                            index = np.unravel_index(i, self.output_size)
                            start_index = tuple([a * b for a, b in zip(index, self.stride_shape)])
                            end_index = tuple([a+b for a, b in zip(start_index, self.kernel_shape)])
                            sector_slice = tuple(
                                [slice(start_index[ind], end_index[ind]) for ind in range(len(start_index))]
                            )
                            sector = img_channel[sector_slice]
                            output[img_index][index][self.filters*channel_index + filter_index] = \
                                np.sum(sector * self.kernels[filter_index])
                            output = apply_activation(self.activation, output)
        return output

    def apply_padding(self, inputs):
        pad = [[0, 0]]  # No padding to the images itself
        for p in self.padding_shape:
            pad.append([p, p])
        pad.append([0, 0])  # No padding to the channel
        return tf.pad(inputs, tf.constant(pad), "CONSTANT", 0, name="Zero Padding")

    def save_tensorboard_checkpoint(self, x, weight_summary, activation_summary, step=None):
        return None     # TODO

    def get_description(self):
        return "ComplexConv"    # TODO

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return ComplexConvolutional(filters=self.filters, kernel_shape=self.kernel_shape, input_shape=self.input_size,
                                    padding=self.padding_shape, stride=self.stride_shape, input_dtype=self.input_dtype)

    def get_real_equivalent(self):
        return self.__deepcopy__()


if __name__ == "__main__":
    conv = ComplexConvolutional(1, (3, 3), (6, 6), padding=0, input_dtype=np.complex64)
    # https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/
    img1 = [
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]
    ]
    img2 = [
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0]
    ]
    img = np.zeros((2, 6, 6, 2))
    img[0, ..., 0] = img1
    img[0, ..., 1] = img2
    img[1, ..., 0] = img1
    img[1, ..., 1] = img2
    conv.kernels[0] = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]
    out1 = conv(img)
    """
    # https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
    img3 = [0, 0, 0, 1, 1, 0, 0, 0]
    img_padd = [
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    ]

    conv = ComplexConvolutional(1, 3, (8, 2, 2), padding=1, input_dtype=np.complex64)
    conv.kernels[0] = [0, 1, 0]
    out3 = conv(img_padd)
    """
    import pdb; pdb.set_trace()


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.0.20'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
