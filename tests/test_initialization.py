import tensorflow as tf
from cvnn import logger
import cvnn.initializers as initializers
from pdb import set_trace
import sys

shape = (3, 3, 3)
dtype = tf.dtypes.float32


def compare(key, tf_init, my_init):
    tf_version = tf_init(seed=100)(shape=shape, dtype=dtype)
    my_version = my_init(seed=100)(shape=shape, dtype=dtype)
    comparison = tf_version.numpy() == my_version.numpy()
    if comparison.all():
        print(f"{key} initialization works fine")
    else:
        logger.error(f"ERROR! FAIL! {key} initialization does not work!")
        # print(comparison)
        print("tensorflow version: " + str(tf_version))
        print("own version: " + str(my_version))
        sys.exit(-1)


tests = {
    "He Uniform": [tf.initializers.he_uniform, initializers.ComplexHeUniform],
    "Glorot Uniform": [tf.initializers.GlorotUniform, initializers.ComplexGlorotUniform],
    "He Normal": [tf.initializers.he_normal, initializers.ComplexHeNormal],
    "Glorot Normal": [tf.initializers.GlorotNormal, initializers.ComplexGlorotNormal]
}

def test_inits():
    for key, value in tests.items():
        compare(key, value[0], value[1])

if __name__ == "__main__":
    test_inits()