import sys
import tensorflow as tf
from tensorflow.keras import datasets
from time import perf_counter
import numpy as np
from pdb import set_trace
import sys

ENABLE_MEMORY_GROWTH = True     # https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
DEBUG_CONV = False
TEST_KERAS_CONV2D = False
TEST_CONV_SPEED = False

if ENABLE_MEMORY_GROWTH:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if TEST_KERAS_CONV2D:    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0    # Normalize pixel values to be between 0 and 1
    start_time = perf_counter()
    conv2d = tf.keras.layers.Conv2D(1, 3, input_shape=(32, 32, 3))
    k_out = conv2d(train_images[:32].astype(np.float32))        
    end_time = perf_counter()
    # Without memory growth: 12.695380546000003; 3.7; 11.6; 13.1
    # With Memory Growth: 1.4; 1.2; 1.15; 4.132; 
    # Failed to initialize GPU device #0: unknown error
    print("Computing time was {} seconds".format(end_time - start_time))
    sys.exit()

class Dense:
    def __init__(self, output_size, input_size, activation=tf.keras.activations.relu):
        self.w = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_size, output_size)))
        self.b = tf.Variable(tf.keras.initializers.Zeros()(shape=output_size))
        self.activation = activation
        self.__class__.__call__ = self.call
        
                             
    def call(self, inputs):
        return self.activation(tf.add(tf.matmul(inputs, self.w), self.b))
    
    def trainable_variables(self):
        return [self.w, self.b]

class Flatten:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = np.prod(self.input_size)
        self.__class__.__call__ = self.call
        
    def call(self, inputs):
        return tf.reshape(inputs, (inputs.shape[0], self.output_size))

    def trainable_variables(self):
        return []

class ConvND:
    def __init__(self, kernels, input_size, kernel_shape=(3, 3), padding=0, stride=1,  activation=tf.keras.activations.linear):
        self.filters = kernels
        self.input_size = input_size
        self.activation = activation
        self._calculate_shapes(kernel_shape, padding, stride)
        self.__class__.last_layer_output_size = self.output_size
        self._init_kernel()
        self.__class__.__call__ = self.call
        
    def _init_kernel(self):
        self.kernels = []
        this_shape = self.kernel_shape + (self.input_size[-1],)
        for _ in range(self.filters):
            self.kernels.append(tf.Variable(tf.keras.initializers.GlorotUniform()(shape=this_shape)))
        self.bias = tf.Variable(tf.keras.initializers.Zeros()(shape=self.filters))
        
    def _calculate_shapes(self, kernel_shape, padding, stride):
        if isinstance(kernel_shape, int):
            self.kernel_shape = (kernel_shape,) * (len(self.input_size) - 1)    # -1 because the last is the channel
        elif isinstance(kernel_shape, (tuple, list)):
            self.kernel_shape = tuple(kernel_shape)
        else:
            print(
                "Kernel shape: " + str(kernel_shape) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        # Padding
        if isinstance(padding, int):
            self.padding_shape = (padding,) * (len(self.input_size) - 1)    # -1 because the last is the channel
            # I call super first in the case input_shape is none
        elif isinstance(padding, (tuple, list)):
            self.padding_shape = tuple(padding)
        else:
            print("Padding: " + str(padding) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        # Stride
        if isinstance(stride, int):
            self.stride_shape = (stride,) * (len(self.input_size) - 1)
            # I call super first in the case input_shape is none
        elif isinstance(stride, (tuple, list)):
            self.stride_shape = tuple(stride)
        else:
            print("stride: " + str(stride) + " format not supported. It must be an int or a tuple")
            sys.exit(-1)
        out_list = []
        for i in range(len(self.input_size) - 1):   # -1 because the number of input channels is irrelevant
            # 2.4 on https://arxiv.org/abs/1603.07285
            out_list.append(int(np.floor(
                (self.input_size[i] + 2 * self.padding_shape[i] - self.kernel_shape[i]) / self.stride_shape[i]
            ) + 1))
        out_list.append(self.filters)       # New channels are actually the filters
        self.output_size = tuple(out_list)
        return self.output_size
    
    def trainable_variables(self):
        return self.kernels + [self.bias]
    
    # @tf.function
    def call(self, inputs):
        inputs = self.apply_padding(inputs)             # Add zeros if needed
        output_np = np.zeros(                           # I use np because tf does not support the assigment
            (inputs.shape[0],) +                        # Per each image
            self.output_size, dtype=np.float32
        )
        img_index = 0
        progbar = tf.keras.utils.Progbar(inputs.shape[0])
        for image in inputs:
            for filter_index in range(self.filters):
                for i in range(int(np.prod(self.output_size[:-1]))):  # for each element in the output
                    index = np.unravel_index(i, self.output_size[:-1])
                    start_index = tuple([a * b for a, b in zip(index, self.stride_shape)])
                    end_index = tuple([a+b for a, b in zip(start_index, self.kernel_shape)])
                    sector_slice = tuple(
                        [slice(start_index[ind], end_index[ind]) for ind in range(len(start_index))]
                    )
                    sector = image[sector_slice]
                    new_value = tf.reduce_sum(sector * self.kernels[filter_index]) + self.bias[filter_index]
                    indices = (img_index,) + index + (filter_index,)
                    mask = tf.Variable(tf.fill(output_np.shape, 1))
                    mask = mask[indices].assign(0)
                    mask = tf.cast(mask, dtype=np.float32)
                    output_np = output_np * mask + (1 - mask) * new_value
                    
                    # import pdb; pdb.set_trace()
            img_index += 1
            progbar.update(img_index)
        output = self.activation(output_np)
        return output

    def apply_padding(self, inputs):
        pad = [[0, 0]]  # No padding to the images itself
        for p in self.padding_shape:
            pad.append([p, p])
        pad.append([0, 0])  # No padding to the channel
        return tf.pad(inputs, tf.constant(pad), "CONSTANT", 0)

# Test conv works: https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/
# set_trace()
# Prepare to test conv layers
"""
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 425.31       Driver Version: 425.31       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 735M    WDDM  | 00000000:01:00.0 N/A |                  N/A |
| N/A   58C    P0    N/A /  N/A |     37MiB /  1024MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0                    Not Supported                                       |
+-----------------------------------------------------------------------------+
"""
if DEBUG_CONV:
    img1 = np.array([
            [3, 0, 1, 2, 7, 4],
            [1, 5, 8, 9, 3, 1],
            [2, 7, 2, 5, 1, 3],
            [0, 1, 3, 1, 7, 8],
            [4, 2, 1, 6, 2, 8],
            [2, 4, 5, 2, 3, 9]
        ]).astype(np.float32)
    img2 = np.array([
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0]
        ]).astype(np.float32)
    img1 = np.reshape(img1, (1, 6, 6, 1))
    img2 = np.reshape(img2, (1, 6, 6, 1))
    conv = ConvND(1, kernel_shape=(3, 3), input_size=(6, 6, 1), padding=0)
    conv.kernels[0] = np.reshape(np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]), (3, 3, 1))
    out1 = conv(img1)
    out2 = conv(img2)
    print(out1[0,...,0])
    print(out2[0,...,0])
# set_trace()
# conv tested
"""
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 425.31       Driver Version: 425.31       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 735M    WDDM  | 00000000:01:00.0 N/A |                  N/A |
| N/A   58C    P0    N/A /  N/A |    110MiB /  1024MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0                    Not Supported                                       |
+-----------------------------------------------------------------------------+
"""
# Model class to train network
class Model:
    def __init__(self, shape):
        self.shape = shape
        self.__class__.__call__ = self.call
        
    def call(self, x):
        for i in range(len(self.shape)):  # Apply all the layers
            x = self.shape[i].call(x)
        return x
        
    def fit(self, x, y, epochs=10, batch_size=32, learning_rate=0.01):
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=batch_size)
        num_tr_iter = int(x.shape[0] / batch_size)
        for epoch in range(epochs):
            iteration = 0
            tf.print("\nEpoch {0}/{1}".format(epoch+1, epochs))
            progbar = tf.keras.utils.Progbar(num_tr_iter)
            for x_batch, y_batch in train_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache():
                progbar.update(iteration)
                iteration += 1
                self._train_step(x_batch, y_batch, learning_rate)
                
    def _apply_loss(self, y_true, y_pred):
        return tf.reduce_mean(input_tensor=tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    
    @tf.function      # This makes all faster but harder to debug (set_trace is broken and print doesn't work)
    def _train_step(self, x_train_batch, y_train_batch, learning_rate):
        with tf.GradientTape() as tape:
            with tf.name_scope("Forward_Phase") as scope:
                tf.print("Forward mode")
                x_called = self.call(x_train_batch)  # Forward mode computation
            # Loss function computation
            with tf.name_scope("Loss") as scope:
                tf.print("Compute loss")
                current_loss = self._apply_loss(y_train_batch, x_called)  # Compute loss

        # Calculating gradient
        with tf.name_scope("Gradient") as scope:
            tf.print("Get trainable variables")
            variables = []
            for lay in self.shape:
                variables.extend(lay.trainable_variables())  # TODO: Debug this for all layers.
            tf.print("Compute gradients")
            gradients = tape.gradient(current_loss, variables)  # Compute gradients
            assert all(g is not None for g in gradients)

        # Backpropagation
        with tf.name_scope("Optimizer") as scope:
            tf.print("Assign values")
            for i, val in enumerate(variables):
                val.assign(val - learning_rate * gradients[i])

# Prepare Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0    # Normalize pixel values to be between 0 and 1

print(train_images.shape)

if TEST_CONV_SPEED:
    start_time = perf_counter()
    conv_layer = ConvND(1, kernel_shape=(3, 3), input_size=(32, 32, 3))
    out = conv_layer(train_images[:32].astype(np.float32))                  # 152x2 secs, 475.65
    end_time = perf_counter()
    print("Computing time was {} seconds".format(end_time - start_time))

"""
I sometimes have:
    Failed to initialize GPU device #0: unknown error

with @tf.function decorator I have the error:
    Failed to initialize GPU device #0: unknown error
    2020-06-23 19:11:09.754024: F .\tensorflow/core/kernels/random_op_gpu.h:227] Non-OK-status: GpuLaunchKernel(FillPhiloxRandomKernelLaunch<Distribution>, num_blocks, block_size, 0, d.stream(), gen, data, size, dist) status: Internal: invalid configuration argument

    or 

    Traceback (most recent call last):
    TypeError: in converted code:
    TypeError: tf__call() takes 2 positional arguments but 3 were given

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 425.31       Driver Version: 425.31       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 735M    WDDM  | 00000000:01:00.0 N/A |                  N/A |
| N/A   65C    P0    N/A /  N/A |    112MiB /  1024MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0                    Not Supported                                       |
+-----------------------------------------------------------------------------+
"""


# Define layers
model_layers = [
    ConvND(1, kernel_shape=(3, 3), input_size=(32, 32, 3)),
    Flatten((30, 30, 1)),
    Dense(64, activation=tf.keras.activations.relu, input_size=900),
    Dense(10, input_size=64, activation=tf.keras.activations.softmax)
]
model = Model(model_layers)

# set_trace()
# Train Model
model.fit(train_images[:1000].astype(np.float32), train_labels[:1000].astype(np.float32), epochs=5, batch_size=32)

"""
Epoch 1/5
 0/31 [..............................] - ETA: 0sForward mode
 1/32 [..............................] - ETA: 10:372020-06-23 19:38:17.893582: W tensorflow/core/common_runtime/bfc_allocator.cc:419] Allocator (GPU_0_bfc) ran out of memory trying to allocate 112.5KiB (rounded to 115200).  Current allocation summary follows.
2020-06-23 19:38:17.930516: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (256):   Total Chunks: 3940, Chunks in use: 3940. 985.0KiB allocated for chunks. 985.0KiB in use in bin. 215.6KiB client-requested in use in bin.
2020-06-23 19:38:17.977848: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (512):   Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.017997: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (1024):  Total Chunks: 1, Chunks in use: 1. 1.3KiB allocated for chunks. 1.3KiB in use in bin. 1.0KiB client-requested in use in bin.
2020-06-23 19:38:18.058818: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (2048):  Total Chunks: 1, Chunks in use: 1. 2.5KiB allocated for chunks. 2.5KiB in use in bin. 2.5KiB client-requested in use in bin.
2020-06-23 19:38:18.100510: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (4096):  Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.141216: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (8192):  Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.176846: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (16384):         Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.207463: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (32768):         Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.242445: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (65536):         Total Chunks: 4908, Chunks in use: 4908. 539.22MiB allocated for chunks. 539.22MiB in use in bin. 539.21MiB client-requested in use in bin.
2020-06-23 19:38:18.280173: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (131072):        Total Chunks: 8, Chunks in use: 8. 1.40MiB allocated for chunks. 1.40MiB in use in bin. 1012.5KiB client-requested in use in bin.
2020-06-23 19:38:18.320035: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (262144):        Total Chunks: 1, Chunks in use: 1. 450.3KiB allocated for chunks. 450.3KiB in use in bin. 384.0KiB client-requested in use in bin.
2020-06-23 19:38:18.376468: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (524288):        Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.413848: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (1048576):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.450537: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (2097152):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.486643: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (4194304):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.523049: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (8388608):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.554221: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (16777216):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.590211: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (33554432):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.625964: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (67108864):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.664359: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (134217728):     Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.701712: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (268435456):     Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-06-23 19:38:18.743990: I tensorflow/core/common_runtime/bfc_allocator.cc:885] Bin for 112.5KiB was 64.0KiB, Chunk State:
2020-06-23 19:38:18.760060: I tensorflow/core/common_runtime/bfc_allocator.cc:898] Next region of size 1048576
2020-06-23 19:38:18.778819: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0000000600F80000 next 1 of size 1280
2020-06-23 19:38:18.797981: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0000000600F80500 next 2 of size 256

.... A very long repetition of this message


2020-06-23 19:42:42.059088: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0000000622D51200 next 8858 of size 256
2020-06-23 19:42:42.077451: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0000000622D51300 next 18446744073709551615 of size 214528
2020-06-23 19:42:42.097273: I tensorflow/core/common_runtime/bfc_allocator.cc:914]      Summary of in-use Chunks by size:
2020-06-23 19:42:42.115466: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3940 Chunks of size 256 totalling 985.0KiB
2020-06-23 19:42:42.132905: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1280 totalling 1.3KiB
2020-06-23 19:42:42.149939: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 2560 totalling 2.5KiB
2020-06-23 19:42:42.167957: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 4905 Chunks of size 115200 totalling 538.88MiB
2020-06-23 19:42:42.185877: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 116224 totalling 113.5KiB
2020-06-23 19:42:42.203308: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 117504 totalling 114.8KiB
2020-06-23 19:42:42.222310: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 119296 totalling 116.5KiB
2020-06-23 19:42:42.244583: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 138752 totalling 135.5KiB
2020-06-23 19:42:42.268096: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 155392 totalling 151.8KiB
2020-06-23 19:42:42.288503: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 158720 totalling 155.0KiB
2020-06-23 19:42:42.307893: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 173312 totalling 169.3KiB
2020-06-23 19:42:42.333169: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 195072 totalling 190.5KiB
2020-06-23 19:42:42.355288: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 202240 totalling 197.5KiB
2020-06-23 19:42:42.376818: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 214528 totalling 209.5KiB
2020-06-23 19:42:42.400296: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 230400 totalling 225.0KiB
2020-06-23 19:42:42.431019: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 461056 totalling 450.3KiB
2020-06-23 19:42:42.449943: I tensorflow/core/common_runtime/bfc_allocator.cc:921] Sum Total of in-use chunks: 542.02MiB
2020-06-23 19:42:42.473861: I tensorflow/core/common_runtime/bfc_allocator.cc:923] total_region_allocated_bytes_: 568350976 memory_limit_: 568351129 available bytes: 153 curr_region_allocation_bytes_: 1073741824
2020-06-23 19:42:42.505584: I tensorflow/core/common_runtime/bfc_allocator.cc:929] Stats:
Limit:                   568351129
InUse:                   568350976
MaxInUse:                568350976
NumAllocs:                   12827
MaxAllocSize:               461056

2020-06-23 19:42:42.543755: W tensorflow/core/common_runtime/bfc_allocator.cc:424] 
2020-06-23 19:42:42.572182: W tensorflow/core/framework/op_kernel.cc:1622] OP_REQUIRES failed at cast_op.cc:109 : Resource exhausted: OOM when allocating tensor with shape[32,30,30,1] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
2020-06-23 19:42:42.620366: W tensorflow/core/kernels/data/cache_dataset_ops.cc:820] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset will be discarded. This can happen if you have an input pipeline similar to dataset.cache().take(k).repeat(). You should use dataset.take(k).cache().repeat() instead.
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[32,30,30,1] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:Cast] name: Forward_Phase/Cast/
"""