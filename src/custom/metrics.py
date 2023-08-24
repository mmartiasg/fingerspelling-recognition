import tensorflow as tf


@tf.keras.utils.register_keras_serializable(name="sparse_levenshtein_v1")
class SparseLevenshtein(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_levenshtein_v1", **kwargs):
        super().__init__(name=name, **kwargs)
        #Sum average
        self.absolute_sum = self.add_weight(name="absolute_sum", initializer="zeros", dtype="float32")
        #Total samples sum batch size
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")
        self.supports_masking = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        distance = tf.edit_distance(tf.sparse.from_dense(y_true), tf.sparse.from_dense(tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.int32)))
        distance_reduce_sum = tf.reduce_sum(distance)

        self.absolute_sum.assign_add(distance_reduce_sum)
        self.total_samples.assign_add(tf.shape(y_pred)[0])

    def result(self):
        return self.absolute_sum/tf.cast(self.total_samples, dtype=tf.float32)

    def reset_state(self):
        self.absolute_sum.assign(0.)
        self.total_samples.assign(0)

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable(name="sparse_levenshtein_v2")
class SparseLevenshteinV2(tf.keras.metrics.Metric):

    def __init__(self, name="sparse_levenshtein_v2", **kwargs):
        super().__init__(name=name, **kwargs)
        #Sum average
        self.absolute_sum = self.add_weight(name="absolute_sum", initializer="zeros", dtype="float32")
        #Total samples sum batch size
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        distance = tf.edit_distance(tf.sparse.from_dense(y_true), tf.sparse.from_dense(tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)))
        distance_reduce_sum = tf.reduce_sum(distance)
        
        self.absolute_sum.assign_add(distance_reduce_sum)
        self.total_samples.assign_add(tf.shape(y_pred)[0])

    def result(self):
        return self.absolute_sum/tf.cast(self.total_samples, dtype=tf.float32)

    def reset_state(self):
        self.absolute_sum.assign(0.)
        self.total_samples.assign(0)

    def get_config(self):
        config = super().get_config()
        return config
