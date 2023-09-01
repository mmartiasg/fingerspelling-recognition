import tensorflow as tf
from src.data_utils.dataset import pad_token_idx

@tf.keras.utils.register_keras_serializable(name="landmarks_embedding_v1")
class LandmarkEmbeddingV1(tf.keras.layers.Layer):
    def __init__(self, embedings_dim=64, max_seq_length=120, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.embedings_dim = embedings_dim
        self.max_seq_length = max_seq_length
        self.kernel_size = kernel_size

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

        return x + self.positional_encoding(tf.range(start=0, limit=self.max_seq_length, delta=1))

    def compute_mask(self, inputs, mask=None):
        # This will return a boolean matrix where is False when found a 0 in that position
        # axis=2 will mask the sequence as a whole. In this case I have a fix number of sequences.
        # return tf.math.not_equal(inputs, 0)
    
        mask = tf.math.equal(tf.reduce_sum(inputs, axis=2), 0)
        return mask

    def get_build_config(self):
        config = super().get_config()
        config.update({"embedings_dim": self.embedings_dim,
                "max_seq_length": self.max_seq_length,
                "kernel_size": self.kernel_size})

        return config


@tf.keras.utils.register_keras_serializable(name="landmarks_embedding_v2")
class LandmarkEmbeddingV2(tf.keras.layers.Layer):
    def __init__(self, embedings_dim=64, max_seq_length=120, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.embedings_dim = embedings_dim
        self.max_seq_length = max_seq_length
        self.kernel_size = kernel_size

        self.conv1 = tf.keras.layers.Conv1D(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1DTranspose(
            embedings_dim, kernel_size, strides=2, padding="same", activation="relu"
        )
        self.positional_encoding = tf.keras.layers.Embedding(input_dim=max_seq_length, output_dim=embedings_dim, mask_zero=True)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)

        return x + self.positional_encoding(tf.range(start=0, limit=self.max_seq_length, delta=1))

    def compute_mask(self, inputs, mask=None):
        # This will return a boolean matrix where is False when found a 0 in that position
        # axis=2 will mask the sequence as a whole. In this case I have a fix number of sequences.
        # return tf.math.not_equal(inputs, 0)
    
        mask = tf.math.equal(tf.reduce_sum(inputs, axis=2), 0)
        return mask

    def get_build_config(self):
        config = super().get_config()
        config.update({"embedings_dim": self.embedings_dim,
                "max_seq_length": self.max_seq_length,
                "kernel_size": self.kernel_size})

        return config


@tf.keras.utils.register_keras_serializable(name="landmarks_embedding_v3")
class LandmarkEmbeddingV3(tf.keras.layers.Layer):
    def __init__(self, embedings_dim=64, max_seq_length=120, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.embedings_dim = embedings_dim
        self.max_seq_length = max_seq_length
        self.kernel_size = kernel_size

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
                "max_seq_length": self.max_seq_length,
                "kernel_size": self.kernel_size}

