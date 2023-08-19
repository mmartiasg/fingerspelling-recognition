import tensorflow as tf


class SparseLevenshtein(tf.keras.metrics.Metric):

    def __init__(self, name="sparse_levenshtein", **kwargs):
        super().__init__(name=name, **kwargs)
        #Sum average
        self.absolute_sum = self.add_weight(name="absolute_sum", initializer="zeros", dtype="float32")
        #Total samples sum batch size
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        distance = tf.reduce_sum(tf.edit_distance(tf.sparse.from_dense(y_true), tf.sparse.from_dense(tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.int32))))
        # distance = tf.clip_by_value(distance, clip_value_min=1.0, clip_value_max=-1.0)
        self.absolute_sum.assign_add(distance)
        self.total_samples.assign_add(tf.shape(y_pred)[0])

    def result(self):
        return self.absolute_sum/tf.cast(self.total_samples, dtype=tf.float32)

    def reset_state(self):
        self.absolute_sum.assign(0.)
        self.total_samples.assign(0)


