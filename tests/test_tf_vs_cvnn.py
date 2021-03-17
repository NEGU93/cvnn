from examples.cifar410_example import test_cifar10
from examples.fashion_mnist_example import test_fashion_mnist
from examples.mnist_dataset_example import test_mnist
from importlib import reload
import os
import tensorflow


def test_tf_vs_cvnn():
    """
    This modules compares cvnn when working with float numbers so that it gives the exact same value as tensorflow.
    """
    reload(tensorflow)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    test_mnist()
    test_fashion_mnist()
    test_cifar10()


if __name__ == '__main__':
    test_tf_vs_cvnn()
