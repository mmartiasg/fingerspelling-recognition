import tensorflow as tf


class LandmarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedings_dim=64, max_seq_length=120, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.embedings_dim = embedings_dim
        self.max_seq_length = max_seq_length

        self.conv1 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1DTranspose(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv4 = tf.keras.layers.Conv1DTranspose(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.positional_encoding = tf.keras.layers.Embedding(input_dim=max_seq_length, output_dim=embedings_dim, mask_zero=True)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        positions = tf.range(start=0, limit=self.max_seq_length, delta=1)
        return x + self.positional_encoding(positions)

    def compute_mask(self, inputs, mask=None):
    #     # This will return a boolean matrix where is False when found a 0 in that position
        # axis=2 will mask the sequence as a whole. In this case I have a fix number of sequences.
    #     return tf.math.not_equal(inputs, 0)
        mask = tf.math.equal(tf.reduce_sum(inputs, axis=2), 0)
        return mask

    def get_build_config(self):
        return {"embedings_dim": self.embedings_dim,
                "max_seq_length": self.max_seq_length}



class LandmarkEmbeddingV2(tf.keras.layers.Layer):
    def __init__(self, embedings_dim=64, max_seq_length=120, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.embedings_dim = embedings_dim
        self.max_seq_length = max_seq_length

        self.conv1 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv4 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def get_build_config(self):
        return {"embedings_dim": self.embedings_dim,
                "max_seq_length": self.max_seq_length}
