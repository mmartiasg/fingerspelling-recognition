import tensorflow as tf

@tf.keras.utils.register_keras_serializable(name="encoder")
class TransformerEncoder(tf.keras.layers.Layer):
    # Do not forget the **kwargs
    def __init__(self, num_heads, dim_emb, dim_dense, **kwargs):
        """
            dim_emb: This is the embedding size
            dim_dense: The size of the dense layers used to project the new embeddings weighted by the attention score
            num_heads: Number of subspaces
        """

        # pass the pointer to super.__init__()!
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.dim_dense = dim_dense

        # TODO : should I omit the value_dim=self.dim_dense?
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim_emb,
                                                      value_dim=self.dim_dense)

        # I use dim_emb at the end to have a projection with the same units as the input
        self.projection = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=self.dim_dense, activation="tanh"), tf.keras.layers.Dense(units=self.dim_emb)])

        self.nomrlayer_1 = tf.keras.layers.LayerNormalization()

        self.nomrlayer_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None, **kwargs):
        """
            The inputs will be batched, sentence_length or None if not limited, dim_embeddings
            :param mask:
            :param inputs:
        """

        # Set the mask for when I need to ignore the tokens used to pad the sentence
        if mask is not None:
        #     # Why this new axis?
        #     # Reduce each sequence to a single vector for classification via a global pooling layer
        #     # I think the first is batch, second will be occupy by the sentence and last by dimensions embeddings
        #     # The mask I get is not complete then?
        #     # shape [batch_size, 1, mask values]
            mask = mask[:, tf.newaxis, :]

        # Attention protections vectors from q k and v
        # in this case Q==K==V self attention
        attention_out = self.mha(query=inputs, key=inputs, value=inputs, attention_mask=mask)

        # Residual connection with a norm layer to scale the values.
        # Without this values could get bigger and that could introduce learning disabilities for the network to learn
        proy_in = self.nomrlayer_1(inputs + attention_out)

        # project the output of the attention layer. This could prove useful to learn representations
        proy_out = self.projection(proy_in)

        # Normalized residual connection, the same as before we avoid 2 problems gradient vanish and gradient
        # explosion by adding a residual connection and a normalization layer
        return self.nomrlayer_2(proy_in + proy_out)

    def get_config(self):
        """
            This is a dictionary with the parameter's values to re-instantiate the layer when the model is loaded
        """

        return {"num_heads": self.num_heads,
                "dim_emb": self.dim_emb,
                "dim_dense": self.dim_dense}
