from examples.cifar410_example import test_cifar10
from examples.fashion_mnist_example import test_fashion_mnist
from examples.mnist_dataset_example import test_mnist


def test_tf_vs_cvnn():
    """
    This modules compares cvnn when working with float numbers so that it gives the exact same value as tensorflow.
    """
    test_cifar10()
    test_fashion_mnist()
    test_mnist()
