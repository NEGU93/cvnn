import tensorflow as tf
import rvnn_v1_compat


def check_tf_version():
    # Makes sure tensorflow version is 2
    assert tf.__version__.startswith('2')


def check_gpu_compatible():
    print("Available GPU devices:", flush=True)
    print(tf.test.gpu_device_name(), flush=True)
    print("Built in with CUDA: " + str(tf.test.is_built_with_cuda()), flush=True)
    print("GPU available: " + str(tf.test.is_gpu_available()), flush=True)


if __name__ == "__main__":
    # check_tf_version()
    # check_gpu_compatible()

    rvnn_v1_compat.rvnn()


