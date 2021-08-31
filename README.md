# Complex-Valued Neural Networks (CVNN)
Done by @NEGU93 - J. Agustin Barrachina

[![Documentation Status](https://readthedocs.org/projects/complex-valued-neural-networks/badge/?version=latest)](https://complex-valued-neural-networks.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/cvnn.svg)](https://badge.fury.io/py/cvnn) [![Anaconda cvnn version](https://img.shields.io/conda/v/NEGU93/cvnn.svg)](https://anaconda.org/negu93/cvnn) [![DOI](https://zenodo.org/badge/296050056.svg)](https://zenodo.org/badge/latestdoi/296050056)

Using this library, the only difference with a Tensorflow code is that you should use `cvnn.layers` module instead of `tf.keras.layers`.

This is a library that uses [Tensorflow](https://www.tensorflow.org) as a back-end to do complex-valued neural networks as CVNNs are barely supported by Tensorflow and not even supported yet for [pytorch](https://github.com/pytorch/pytorch/issues/755) (reason why I decided to use Tensorflow for this library). To the authors knowledge, **this is the first library that actually works with complex data types** instead of real value vectors that are interpreted as real and imaginary part.

Update:
  - Since [v1.6](https://pytorch.org/blog/pytorch-1.6-released/#beta-complex-numbers) (28 July 2020), pytorch now supports complex vectors and complex gradient as BETA. But still have the same issues that Tensorflow has, so no reason to migrate yet.
  - Since [v0.2](https://github.com/wavefrontshaping/complexPyTorch/releases/tag/0.2) (25 Jan 2021) [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch) uses complex64 dtype.

## Documentation

Please [Read the Docs](https://complex-valued-neural-networks.readthedocs.io/en/latest/index.html)

## Instalation Guide:

Using [Anaconda](https://anaconda.org/negu93/cvnn)

```
conda install -c negu93 cvnn
```

Using [PIP](https://pypi.org/project/cvnn/)

**Vanilla Version**
installs all the minimum dependencies.

```
pip install cvnn
```
**Plot capabilities**
has the posibility to plot the results obtained with the training with several plot libraries.

```
pip install cvnn[plotter]
```

**Full Version** installs full version with all features

```
pip install cvnn[full]
```

## Short example

From "outside" everything is the same as when using Tensorflow.

```
import numpy as np
import tensorflow as tf

# Assume you already have complex data... example numpy arrays of dtype np.complex64
(train_images, train_labels), (test_images, test_labels) = get_dataset()        # to be done by each user

model = get_model()   # Get your model

# Compile as any TensorFlow model
model.compile(optimizer='adam', metrics=['accuracy'],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()

# Train and evaluate
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```
The main difference is that you will be using `cvnn` layers instead of Tensorflow layers.
There are some options on how to do it as shown here:

### Sequential API
```
import cvnn.layers as complex_layers

def get_model():
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(32, 32, 3)))                     # Always use ComplexInput at the start
    model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
    model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(64, activation='cart_relu'))
    model.add(complex_layers.ComplexDense(10, activation='convert_to_real_with_abs'))   
    # An activation that casts to real must be used at the last layer. 
    # The loss function cannot minimize a complex number
    return model
```
### Functional API
```
import cvnn.layers as complex_layers
def get_model():
    inputs = complex_layers.complex_input(shape=(128, 128, 3))
    c0 = complex_layers.ComplexConv2D(32, activation='cart_relu', kernel_size=3)(inputs)
    c1 = complex_layers.ComplexConv2D(32, activation='cart_relu', kernel_size=3)(c0)
    c2 = complex_layers.ComplexMaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)
    t01 = complex_layers.ComplexConv2DTranspose(5, kernel_size=2, strides=(2, 2), activation='cart_relu')(c2)
    concat01 = tf.keras.layers.concatenate([t01, c1], axis=-1)

    c3 = complex_layers.ComplexConv2D(4, activation='cart_relu', kernel_size=3)(concat01)
    out = complex_layers.ComplexConv2D(4, activation='cart_relu', kernel_size=3)(c3)
    return tf.keras.Model(inputs, out)
```


## About me & Motivation

[My personal website](https://negu93.github.io/agustinbarrachina/)

I am a PhD student from [Ecole CentraleSupelec](https://www.centralesupelec.fr/)
with a scholarship from [ONERA](https://www.onera.fr/en) and the [DGA](https://www.defense.gouv.fr/dga)

I am basically working with Complex-Valued Neural Networks for my PhD topic.
In the need of making my coding more dynamic I build a library not to have to repeat the same code over and over for little changes and accelerate therefore my coding.

## Cite Me

Alway prefer the [Zenodo](https://zenodo.org/record/4452131/export/hx#.YAkuw-j0mUl) citation. 

Next you have a model but beware to change the version and date accordingly.

```
@software{j_agustin_barrachina_2021_4452131,
  author       = {J Agustin Barrachina},
  title        = {Complex-Valued Neural Networks (CVNN)},
  month        = jan,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.3},
  doi          = {10.5281/zenodo.4452131},
  url          = {https://doi.org/10.5281/zenodo.4452131}
}
```

## Issues

For any issues please report them in [here](https://github.com/NEGU93/cvnn/issues)

This library is tested using [pytest](https://docs.pytest.org/).

![pytest logo](tests/pytest.png)



