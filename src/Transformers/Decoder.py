import tensorflow as tf

@tf.keras.utils.register_keras_serializable(name="decoder")
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_emb, dim_dense, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.dim_dense = dim_dense

        self.mha_1 = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim_emb,
                                                        value_dim=self.dim_dense)
        self.mha_2 = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim_emb,
                                                        value_dim=self.dim_dense)

        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.norm_3 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True
        self.proy = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=self.dim_dense, activation="relu"), tf.keras.layers.Dense(units=self.dim_emb)])

    def get_casual_attention_mask(self, inputs):
        """
        :return:
        :param inputs:
        :return:
        """
        # Is this the vectorized batch?
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]

        # Add a new dim to the range this will be
        # [sequence_length, 1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        # [sequence_length,]
        j = tf.range(sequence_length)

        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            # Add a dimension [batch size, 1]
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )

        return tf.tile(mask, mult)

    def call(self, source_encoded, target_input, mask=None):
        """

        :type source_encoded: object
        :type target_input: object
        """
        # set casual attention mask
        # lower diagonal matrix [MAX_SEQ, MAX_SEQ]
        casual_mask = self.get_casual_attention_mask(target_input)

        padding_mask = None
        # Set the mask for when I need to ignore the tokens used to pad the sentence
        if mask is not None:
            # Why this new axis?
            # Reduce each sequence to a single vector for classification via a global pooling layer
            # Still do not get it :C TODO: review
            # I think the first is batch, second will be occupy by the sentence and last by dimensions embeddings
            # The mask I get is not complete then?

            # SHAPE [batch-size, 1, MAX_SEQ]
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")

            # This is to limit the causal mask to the sequence padding have the same shape by row to be able to
            # broadcast it over
            padding_mask = tf.minimum(padding_mask, casual_mask)

        # Self attention over the sequence we want to decode the source
        attention_1_output = self.mha_1(query=target_input, key=target_input, value=target_input,
                                        attention_mask=casual_mask)
        nomr_1_output = self.norm_1(target_input + attention_1_output)

        # Attention between the source encoded and the encoded target
        # result is the weighted target(Q) over the pairwise between the source encoded(V) and the target(K)
        attention_2_output = self.mha_2(query=nomr_1_output, key=source_encoded, value=source_encoded,
                                        attention_mask=padding_mask)

        norm_2_output = self.norm_2(attention_2_output + attention_1_output)

        proy = self.proy(norm_2_output)

        nomr_3_output = self.norm_3(proy + norm_2_output)

        return nomr_3_output

    def get_config(self):
        """
            This is a dictionary with the parameter's values to re-instantiate the layer when the model is loaded
        """
        return {"num_heads": self.num_heads,
                "dim_emb": self.dim_emb,
                "dim_dense": self.dim_dense}
