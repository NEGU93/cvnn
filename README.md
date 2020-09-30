# Complex-Valued Neural Networks (CVNN)
Done by @NEGU93 - J. Agustin Barrachina

[![Documentation Status](https://readthedocs.org/projects/complex-valued-neural-networks/badge/?version=latest)](https://complex-valued-neural-networks.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/cvnn.svg)](https://badge.fury.io/py/cvnn)

This is a library that uses [Tensorflow](https://www.tensorflow.org) as a back-end to do complex-valued neural networks as, as far as I know, CVNNs are barely supported by Tensorflow and not even supported yet for [pytorch](https://github.com/pytorch/pytorch/issues/755) (reason why I decided to use Tensorflow for this library).

Please [Read the Docs](https://complex-valued-neural-networks.readthedocs.io/en/latest/index.html)

## Instalation Guide:

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
    journal      = {GitHub repository},
    year         = {2019}
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
