from tensorflow.keras.callbacks import TensorBoard
from tensorflow import GradientTape
import tensorflow as tf


# This extends TensorBoard to save gradients as histogram
# ExtendedTensorBoard can then be used in replace of tf.keras.callbacks.TensorBoard.
class ExtendedTensorBoard(TensorBoard):
    def _log_gradients(self, epoch):
        writer = self._writers['train']

        with writer.as_default(), GradientTape() as g:
            # here we use test data to calculate the gradients
            features, y_true = list(dataset.batch(100).take(1))[0]

            y_pred = self.model(features)  # forward-propagation
            loss = self.model.compiled_loss(y_true=y_true, y_pred=y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)
