import tensorflow as tf
# https://www.tensorflow.org/guide/autodiff

x = tf.Variable(tf.complex([2., 2.], [2., 2.]))

with tf.GradientTape() as tape:
    y = tf.abs(tf.reduce_sum(x))**2
    print(y)

dy_dx = tape.gradient(y, x)
print(dy_dx)
