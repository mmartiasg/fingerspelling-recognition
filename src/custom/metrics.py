import tensorflow as tf


class Levenshtein(tf.keras.metrics.Metric):

    def __init__(self, name="levenshtein", **kwargs):
        super().__init__(name=name, **kwargs)
        #Sum average
        self.absolute_sum = self.add_weight(name="mse_sum", initializer="zeros")

        #Total samples sum batch size
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        distance = tf.reduce_sum(1 - tf.edit_distance(tf.sparse.from_dense(y_true), tf.sparse.from_dense(y_pred)))
        self.absolute_sum.assign_add(distance)
        self.total_samples.assign_add(y_true.shape[0])

    def result(self):
        return self.absolute_sum/tf.cast(self.total_samples, dtype=tf.float32)
