import tensorflow as tf
import tensorflow_datasets as tfds
from cvnn import layers
import numpy as np
from tqdm import tqdm
from pdb import set_trace
import timeit
try:
    import plotly.graph_objects as go
    import plotly
    PLOTLY = True
except ModuleNotFoundError:
    PLOTLY = False

# tf.enable_v2_behavior()
# tfds.disable_progress_bar()

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'editable': True
}


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def keras_fit(ds_train, ds_test, verbose=True, init1='glorot_uniform', init2='glorot_uniform'):
    # https://www.tensorflow.org/datasets/keras_example
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init1),
      tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=init2)
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    start = timeit.default_timer()
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    stop = timeit.default_timer()
    return model.evaluate(ds_test, verbose=verbose), stop - start


def own_fit(ds_train, ds_test, verbose=True, init1='glorot_uniform', init2='glorot_uniform'):
    model = tf.keras.models.Sequential([
        layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32),
        layers.ComplexDense(128, activation='cart_relu', dtype=np.float32, kernel_initializer=init1),
        layers.ComplexDense(10, activation='softmax_real', dtype=np.float32, kernel_initializer=init2)
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    start = timeit.default_timer()
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    stop = timeit.default_timer()
    return model.evaluate(ds_test, verbose=verbose), stop - start


def test_mnist_montecarlo():
    # Parameters
    KERAS_DEBUG = True
    OWN_MODEL = True
    iterations = 1000
    verbose = 1 if iterations == 1 else 0

    # Training
    ds_train, ds_test = get_dataset()
    keras_results = []
    own_results = []
    keras_fit_time = []
    own_fit_time = []
    if PLOTLY:
        fig = go.Figure()
        fig_time = go.Figure()

    if OWN_MODEL:
        tf.print("Cvnn simulation")
        for _ in tqdm(range(iterations)):
            result = own_fit(ds_train, ds_test, verbose=verbose)
            own_results.append(result[0][1])
            own_fit_time.append(result[-1])
        if PLOTLY:
            fig.add_trace(go.Box(y=own_results, name='Cvnn'))
            fig_time.add_trace(go.Box(y=own_fit_time, name='Cvnn'))
    if KERAS_DEBUG:
        tf.print("Keras simulation")
        for _ in tqdm(range(iterations)):
            result = keras_fit(ds_train, ds_test, verbose=verbose)
            keras_results.append(result[0][1])
            keras_fit_time.append(result[-1])
        if PLOTLY:  
            fig.add_trace(go.Box(y=keras_results, name='Keras'))
            fig_time.add_trace(go.Box(y=keras_fit_time, name='Keras'))
    if PLOTLY:
        plotly.offline.plot(fig, filename="./results/mnist_test.html",
                            config=PLOTLY_CONFIG, auto_open=True)
        plotly.offline.plot(fig_time, filename="./results/mnist_test_fit_time.html",
                            config=PLOTLY_CONFIG, auto_open=True)
    own_results = np.array(own_results)
    keras_results = np.array(keras_results)
    np.save("./results/keras_mnist_test.npy", keras_results)
    np.save("./results/own_mnist_test.npy", own_results)
    own_err = own_results.std() * 2.576 / np.sqrt(50)
    own_mean = own_results.mean()
    keras_err = keras_results.std() * 2.576 / np.sqrt(50)
    keras_mean = keras_results.mean()
    q75, q25 = np.percentile(own_results, [75, 25])
    own_median_err = 1.57 * (q75 - q25) / np.sqrt(50)
    own_median = np.median(own_results)
    q75, q25 = np.percentile(keras_results, [75, 25])
    keras_median_err = 1.57 * (q75 - q25) / np.sqrt(50)
    keras_median = np.median(keras_results)
    mean_str = f"Own Mean: {own_mean * 100} +- {own_err * 100}\nKeras Mean: {keras_mean * 100} +- {keras_err * 100}\n"
    median_str = f"Own Median: {own_median * 100} +- {own_median_err * 100}\nKeras Median: {keras_median * 100} +- {keras_median_err * 100}\n"
    f = open("rmsprop_results.txt", "w+")
    f.write(mean_str)
    f.write(median_str)
    f.close()


def test_mnist():
    seed = 117
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    init1 = tf.constant_initializer(init((784, 128)).numpy())
    init2 = tf.constant_initializer(init((128, 10)).numpy())
    ds_train, ds_test = get_dataset()
    keras1 = keras_fit(ds_train, ds_test, init1=init1, init2=init2)
    keras2 = keras_fit(ds_train, ds_test, init1=init1, init2=init2)
    own = own_fit(ds_train, ds_test, init1=init1, init2=init2)
    print(keras1)
    print(keras2)
    print(own)
    # set_trace()
    
def test_fashion_mnist():
    

if __name__ == "__main__":
    test_mnist()
    # test_mnist_montecarlo()
    # ds_train, ds_test = get_dataset()
    # keras_fit(ds_train, ds_test)
    # own_fit(ds_train, ds_test)


