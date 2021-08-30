import tensorflow as tf
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


def _upsample_tf(in1, in2, units, crop):
    t01 = tf.keras.layers.Conv2DTranspose(units, kernel_size=2, strides=(2, 2), activation='relu')(in1)
    crop01 = tf.keras.layers.Cropping2D(cropping=(crop, crop))(in2)

    concat01 = tf.keras.layers.concatenate([t01, crop01], axis=-1)

    out1 = tf.keras.layers.Conv2D(units, activation='relu', kernel_size=3)(concat01)
    out2 = tf.keras.layers.Conv2D(units, activation='relu', kernel_size=3)(out1)
    return out1, out2


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], INPUT_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], MASK_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_dataset():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    train_length = info.splits['train'].num_examples
    steps_per_epoch = train_length // BATCH_SIZE
    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = train_images.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_batches = test_images.batch(BATCH_SIZE)
    return train_batches, test_batches


def get_tf_model():
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

    outputs = tf.keras.layers.Conv2D(2, kernel_size=1)(c21)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-net-tf")
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == '__main__':
    train_batches, test_batches = get_dataset()
    model = get_tf_model()
    # set_trace()
    model.fit(train_batches, epochs=2, validation_data=test_batches)
