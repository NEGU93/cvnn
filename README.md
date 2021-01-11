# Complex-Valued Neural Networks (CVNN)
Done by @NEGU93 - J. Agustin Barrachina

[![Documentation Status](https://readthedocs.org/projects/complex-valued-neural-networks/badge/?version=latest)](https://complex-valued-neural-networks.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/cvnn.svg)](https://badge.fury.io/py/cvnn) [![Anaconda version](
https://anaconda.org/negu93/cvnn/badges/version.svg)](https://anaconda.org/negu93/cvnn/badges/version.svg) [![DOI](https://zenodo.org/badge/296050056.svg)](https://zenodo.org/badge/latestdoi/296050056)

This is a library that uses [Tensorflow](https://www.tensorflow.org) as a back-end to do complex-valued neural networks as CVNNs are barely supported by Tensorflow and not even supported yet for [pytorch](https://github.com/pytorch/pytorch/issues/755) (reason why I decided to use Tensorflow for this library).

Update:
  - Since v1.6 July 2020, pytorch now supports complex vectors and complex gradient as BETA [source](https://pytorch.org/blog/pytorch-1.6-released/#beta-complex-numbers). But still have the same issues that Tensorflow has, so no reason to migrte yet.

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

```
import numpy as np
import cvnn.layers as complex_layers
import tensorflow as tf

# Assume you already have complex data...
(train_images, train_labels), (test_images, test_labels) = get_dataset()        # to be done by each user

# Create your model
model = models.Sequential()
model.add(complex_layers.ComplexInput(input_shape=(32, 32, 3)))                     # Always use ComplexInput at the start
model.add(complex_layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
model.add(complex_layers.ComplexAvgPooling2D((2, 2)))
model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
model.add(complex_layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(64, activation='cart_relu'))
model.add(complex_layers.ComplexDense(10, activation='convert_to_real_with_abs'))   # An activation that casts to real must be used

# Compile it
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train and evaluate
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```

## About me & Motivation

[My personal website](https://negu93.github.io/agustinbarrachina/)

I am a PhD student from [Ecole CentraleSupelec](https://www.centralesupelec.fr/)
with a scholarship from [ONERA](https://www.onera.fr/en) and the [DGA](https://www.defense.gouv.fr/dga)

I am basically working with Complex-Valued Neural Networks for my PhD topic.
In the need of making my coding more dynamic I build a library not to have to repeat the same code over and over for little changes and accelerate therefore my coding.

## Cite Me

Code:
```
@MISC {NEGU93-CVNN,
    author       = {J. Agustin Barrachina},
    title        = {Complex-Valued Neural Networks (CVNN)},
    howpublished = {\url{https://github.com/NEGU93/cvnn}},
    doi          = {10.5281/zenodo.4140245},
    journal      = {GitHub repository},
    year         = {2019}
}
```
I recommend changing the year and adding a `version` item with the one you used.

Or cite the Zenodo version:
```
@software{j_agustin_barrachina_2020_4140245,
  author       = {J Agustin Barrachina},
  title        = {NEGU93/cvnn: First official release},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.40},
  doi          = {10.5281/zenodo.4140245},
  url          = {https://doi.org/10.5281/zenodo.4140245}
}
```

Paper:
```
@misc{barrachina2020complexvalued,
    title={Complex-Valued vs. Real-Valued Neural Networks for Classification Perspectives: An Example on Non-Circular Data},
    author={Jose Agustin Barrachina and Chenfang Ren and Christele Morisseau and Gilles Vieillard and Jean-Philippe Ovarlez},
    year={2020},
    eprint={2009.08340},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
