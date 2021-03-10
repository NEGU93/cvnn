import tensorflow as tf
import numpy as np
from pdb import set_trace

BATCH_SIZE = 10


def get_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def get_model(init1='glorot_uniform', init2='glorot_uniform'):
    tf.random.set_seed(1)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init1),
        tf.keras.layers.Dense(10, kernel_initializer=init2)
    ])
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def train(model, x_fit, y_fit):
    np.save("initial_weights.npy", np.array(model.get_weights()))
    with tf.GradientTape() as g:
        y_pred = model(x_fit)
        loss = tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y_fit)
        np.save("loss.npy", np.array(loss))
        gradients = g.gradient(loss, model.trainable_weights)
    np.save("gradients.npy", np.array(gradients))
    model.fit(x_fit, y_fit, epochs=1, batch_size=BATCH_SIZE)
    np.save("final_weights.npy", np.array(model.get_weights()))


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = get_dataset()
    model = get_model()
    y_fit = np.zeros((BATCH_SIZE, 10))
    for i, val in enumerate(train_labels[:BATCH_SIZE]):
        y_fit[i][val] = 1.
    train(model, train_images[:BATCH_SIZE], y_fit)
    results = {
        "loss": np.load("loss.npy", allow_pickle=True),
        "init_weights": np.load("initial_weights.npy", allow_pickle=True),
        "gradients": np.load("gradients.npy", allow_pickle=True),
        "final_weights": np.load("final_weights.npy", allow_pickle=True)
    }
    for i_w, f_w, gr in zip(results["init_weights"], results["final_weights"], results["gradients"]):
        gr = gr.numpy()
        print(np.allclose(gr, (i_w - f_w) * BATCH_SIZE / 0.01))
    set_trace()
