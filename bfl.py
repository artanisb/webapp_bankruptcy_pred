import tensorflow as tf
import keras

# Wrapper-Klasse zur Berechnung und Implementierung der Binary Focal Loss-Function
@keras.saving.register_keras_serializable()
class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=4.0, alpha=0.75, name="binary_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)                                            
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow((1 - p_t), self.gamma)
        loss = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "name": self.name
        }