import tensorflow as tf
import numpy as np
from src.data_utils.dataset import pad_token_idx


@tf.keras.utils.register_keras_serializable(name="positional_embedding")
class BasicPositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, dim_emb, max_tokens, max_seq_length, **kwargs):
        super().__init__(**kwargs)

        self.dim_emb = dim_emb
        self.max_tokens = max_tokens
        self.max_seq_length = max_seq_length
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=self.max_tokens, output_dim=self.dim_emb, mask_zero=True)
        # Do I need to set mask_zero True?
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=self.max_seq_length, output_dim=self.dim_emb, mask_zero=True)

    def call(self, inputs, **kwargs):
        # Is this the same as using self.max_seq_length. Although
        lenght = tf.shape(inputs)[-1]

        # this is the sequence information for each token that the network will need to learn
        ordinal_positions = tf.range(start=0, limit=lenght, delta=1)

        # We add to the token embeddings the position information we can do that as those will belong to the same semantic space
        return self.token_embeddings(inputs) + self.position_embeddings(ordinal_positions)

    def compute_mask(self, inputs, mask=None):
        # This will return a boolean matrix where is False when found a pad_token_idx in that position
        return tf.math.not_equal(inputs, pad_token_idx)

    def get_build_config(self):
        config = super().get_config()
        config.update({"dim_emb": self.dim_emb,
                "max_tokens": self.max_tokens,
                "max_seq_length": self.max_seq_length})
        return config
