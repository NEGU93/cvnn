import tensorflow as tf
from cvnn import layers
from pdb import set_trace
import tensorflow_datasets as tfds

BATCH_SIZE = 64
BUFFER_SIZE = 1000
INPUT_SIZE = (572, 572)
MASK_SIZE = (388, 388)


def _downsample_tf(inputs, units):
    c0 = tf.keras.layers.Conv2D(units, activation='relu', kernel_size=3)(inputs)
    c1 = tf.keras.layers.Conv2D(units, activation='relu', kernel_size=3)(c0)
    c2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)
    return c0, c1, c2


def _downsample_cvnn(inputs, units, dtype=tf.float32):
    c0 = layers.ComplexConv2D(units, activation='cart_relu', kernel_size=3, dtype=dtype)(inputs)
    c1 = layers.ComplexConv2D(units, activation='cart_relu', kernel_size=3, dtype=dtype)(c0)
    c2 = layers.ComplexMaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', dtype=dtype)(c1)
    return c0, c1, c2


def _upsample_tf(in1, in2, units, crop):
    t01 = tf.keras.layers.Conv2DTranspose(units, kernel_size=2, strides=(2, 2), activation='relu')(in1)
    crop01 = tf.keras.layers.Cropping2D(cropping=(crop, crop))(in2)

    concat01 = tf.keras.layers.concatenate([t01, crop01], axis=-1)

    out1 = tf.keras.layers.Conv2D(units, activation='relu', kernel_size=3)(concat01)
    out2 = tf.keras.layers.Conv2D(units, activation='relu', kernel_size=3)(out1)
    return out1, out2


def _upsample_cvnn(in1, in2, units, crop, dtype=tf.float32):
    t01 = layers.ComplexConv2DTranspose(units, kernel_size=2, strides=(2, 2), activation='relu', dtype=dtype)(in1)
    crop01 = tf.keras.layers.Cropping2D(cropping=(crop, crop))(in2)

    concat01 = tf.keras.layers.concatenate([t01, crop01], axis=-1)

    out1 = layers.ComplexConv2D(units, activation='relu', kernel_size=3, dtype=dtype)(concat01)
    out2 = layers.ComplexConv2D(units, activation='relu', kernel_size=3, dtype=dtype)(out1)
    return out1, out2


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize_with_pad(datapoint['image'], INPUT_SIZE[0], INPUT_SIZE[1])
    input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], MASK_SIZE[0], MASK_SIZE[1])

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_dataset():
    (train_images, test_images), info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True,
                                                  split=['train[:1%]', 'test[:1%]'])
    train_length = info.splits['train'].num_examples
    steps_per_epoch = train_length // BATCH_SIZE
    train_images = train_images.map(load_image)
    test_images = test_images.map(load_image)

    train_batches = train_images.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_batches = test_images.batch(BATCH_SIZE)
    # set_trace()
    return train_batches, test_batches


def get_cvnn_model(dtype=tf.float32):
    tf.random.set_seed(1)
    inputs = layers.complex_input(shape=INPUT_SIZE + (3,), dtype=dtype)
    # inputs = tf.keras.layers.InputLayer(input_shape=INPUT_SIZE + (3,), dtype=dtype)
    # inputs = tf.keras.layers.Input(shape=INPUT_SIZE + (3,))

    c0, c1, c2 = _downsample_cvnn(inputs, 64, dtype)
    c3, c4, c5 = _downsample_cvnn(c2, 128, dtype)
    c6, c7, c8 = _downsample_cvnn(c5, 256, dtype)
    c9, c10, c11 = _downsample_cvnn(c8, 512, dtype)

    c12 = layers.ComplexConv2D(1024, activation='relu', kernel_size=3, dtype=dtype)(c11)
    c13 = layers.ComplexConv2D(1024, activation='relu', kernel_size=3, padding='valid', dtype=dtype)(c12)

    c14, c15 = _upsample_cvnn(c13, c10, 512, 4, dtype)
    c16, c17 = _upsample_cvnn(c15, c7, 256, 16, dtype)
    c18, c19 = _upsample_cvnn(c17, c4, 128, 40, dtype)
    c20, c21 = _upsample_cvnn(c19, c1, 64, 88, dtype)

    outputs = layers.ComplexConv2D(4, kernel_size=1, dtype=dtype)(c21)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net-cvnn")
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])
    return model


def get_tf_model():
    tf.random.set_seed(1)
    inputs = tf.keras.layers.Input(shape=INPUT_SIZE + (3,))

    c0, c1, c2 = _downsample_tf(inputs, 64)
    c3, c4, c5 = _downsample_tf(c2, 128)
    c6, c7, c8 = _downsample_tf(c5, 256)
    c9, c10, c11 = _downsample_tf(c8, 512)

    c12 = tf.keras.layers.Conv2D(1024, activation='relu', kernel_size=3)(c11)
    c13 = tf.keras.layers.Conv2D(1024, activation='relu', kernel_size=3, padding='valid')(c12)

    c14, c15 = _upsample_tf(c13, c10, 512, 4)
    c16, c17 = _upsample_tf(c15, c7, 256, 16)
    c18, c19 = _upsample_tf(c17, c4, 128, 40)
    c20, c21 = _upsample_tf(c19, c1, 64, 88)

    outputs = tf.keras.layers.Conv2D(4, kernel_size=1)(c21)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net-tf")
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer="adam", metrics=["accuracy"])
    return model


def test_model(model, train_batches, test_batches):
    weigths = model.get_weights()
    # with tf.GradientTape() as tape:
    #     # for elem, label in iter(ds_train):
    #     loss = model.compiled_loss(y_true=tf.convert_to_tensor(test_labels), y_pred=model(test_images))
    #     gradients = tape.gradient(loss, model.trainable_weights)  # back-propagation
    logs = {
        'weights': weigths,
        # 'loss': loss,
        # 'gradients': gradients
    }

    history = model.fit(train_batches, epochs=2, validation_data=test_batches)
    return history, logs


def test_unet():
    train_batches, test_batches = get_dataset()
    history_own, logs_own = test_model(get_cvnn_model(), train_batches, test_batches)
    history_keras, logs_keras = test_model(get_tf_model(), train_batches, test_batches)
    assert history_keras.history == history_own.history, f"\n{history_keras.history}\n !=\n{history_own.history}"


if __name__ == '__main__':
    from importlib import reload
    import os
    import tensorflow

    reload(tensorflow)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    test_unet()
