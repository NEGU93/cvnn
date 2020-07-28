import tensorflow as tf


def make_variables(k, initializer):
    return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
            tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))


v1, v2 = make_variables(3, tf.initializers.GlorotNormal())
print(v1)
print(v2)
