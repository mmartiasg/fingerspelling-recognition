import tensorflow as tf
from src.Transformers.Encoder import TransformerEncoder
from src.Transformers.Decoder import TransformerDecoder
import math
from src.custom.layers import LandmarkEmbeddingV1, LandmarkEmbeddingV2
from src.Transformers.PositionalEncoding import BasicPositionalEmbeddings
from src.constants import TARGET_MAX_LENGHT, MAX_LENGHT_SOURCE
from src.data_utils.dataset import VOCAB_SIZE, start_token_idx, FEATURE_COLUMNS


@tf.keras.utils.register_keras_serializable(name="finger_spelling_v1")
class FingerSpellingV1(tf.keras.Model):
    def __init__(self, attention_heads, embedding_dims, encodder_kernel_size,
                 dense_layers_number, decoded_dropout_rate, output_dropout_rate, max_source_length,
                 max_target_lenght, vocab_size, feature_columns, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention_heads = attention_heads
        self.embedding_dims = embedding_dims
        self.encodder_kernel_size = encodder_kernel_size
        self.dense_layers_number = dense_layers_number
        self.decoded_dropout_rate = decoded_dropout_rate
        self.output_dropout_rate = output_dropout_rate
        self.max_source_length = max_source_length
        self.max_target_lenght = max_target_lenght
        self.vocab_size = vocab_size
        self.feature_columns = feature_columns

        self.source_embeddings = LandmarkEmbeddingV1(embedings_dim=embedding_dims, max_seq_length=max_source_length, kernel_size=encodder_kernel_size)
        self.target_embeddings = BasicPositionalEmbeddings(dim_emb=embedding_dims, max_seq_length=max_target_lenght, max_tokens=vocab_size)
        self.encoder = TransformerEncoder(num_heads=attention_heads, dim_emb=embedding_dims, dim_dense=math.ceil(attention_heads/embedding_dims))
        self.decoder = TransformerDecoder(num_heads=attention_heads, dim_dense=math.ceil(attention_heads/embedding_dims), dim_emb=embedding_dims)

        self.dropout_after_decoder = tf.keras.layers.Dropout(rate=decoded_dropout_rate)
        # keep non zero values
        #Taking hands only x y and just the dominant hand at frame that is why the /2
        self.ffc = tf.keras.Sequential(
                    [tf.keras.layers.Dense(units=int(feature_columns.shape[0]/2), name=f"fccl_{layer_number+1}") for layer_number in range(dense_layers_number)]
                )
        self.dropout_before_output = tf.keras.layers.Dropout(rate=output_dropout_rate)
        self.next_token_proba = tf.keras.layers.Dense(units=vocab_size, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        source_batch = inputs[0]
        target_batch = inputs[1]

        source_embeddings = self.source_embeddings(source_batch)
        encoded_source = self.encoder(source_embeddings)

        target_embeddings = self.target_embeddings(target_batch)
        decoded_sequence = self.decoder(encoded_source, target_embeddings)

        decoded_sequence = self.dropout_after_decoder(decoded_sequence)
        decoded_sequence = self.ffc(decoded_sequence)
        decoded_sequence = self.dropout_before_output(decoded_sequence)

        return self.next_token_proba(decoded_sequence)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, MAX_LENGHT_SOURCE, int(FEATURE_COLUMNS.shape[0]/2)], dtype=tf.float32, name='inputs')])
    def generate_sequence(self, source_sequence_batch):
        batch_size = tf.shape(source_sequence_batch)[0]
        max_lenght = self.max_target_lenght
        target_sequence = tf.ones((batch_size, 1), dtype=tf.int32) * start_token_idx
        for i in range(max_lenght - 1):
            y_preds = self((source_sequence_batch, target_sequence), training=False)
            next_tokens = tf.cast(tf.argmax(y_preds, axis=-1), dtype=tf.int32)
            last_tokens = next_tokens[:, -1][..., tf.newaxis]
            target_sequence = tf.concat([target_sequence, last_tokens], axis=-1)
        return target_sequence

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, MAX_LENGHT_SOURCE, int(FEATURE_COLUMNS.shape[0]/2)], dtype=tf.float32, name='inputs')])
    def generate(self, source):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        landkaremb = self.source_embeddings(source)
        enc = self.encoder(landkaremb, training = False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * start_token_idx
        dec_logits = []
        for i in range(self.max_target_lenght - 1):
            target_emb = self.target_embeddings(dec_input)
            dec_out = self.decoder(enc, target_emb, training = False)
            logits = self.next_token_proba(self.ffc(dec_out), training=False)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

    def get_config(self):
        config = super().get_config()
        config.update({"attention_heads": self.attention_heads,
                "embedding_dims": self.embedding_dims,
                "encodder_kernel_size": self.encodder_kernel_size,
                "dense_layers_number": self.dense_layers_number,
                "decoded_dropout_rate": self.decoded_dropout_rate,
                "output_dropout_rate": self.output_dropout_rate,
                "max_source_length": self.max_source_length,
                "max_target_lenght": self.max_target_lenght,
                "vocab_size": self.vocab_size,
                "feature_columns": self.feature_columns})
        return config


@tf.keras.utils.register_keras_serializable(name="finger_spelling_v2")
class FingerSpellingV2(tf.keras.Model):
    def __init__(self, attention_heads, embedding_dims, encodder_kernel_size,
                 dense_layers_number, decoded_dropout_rate, output_dropout_rate, max_source_length,
                 max_target_lenght, vocab_size, feature_columns, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention_heads = attention_heads
        self.embedding_dims = embedding_dims
        self.encodder_kernel_size = encodder_kernel_size
        self.dense_layers_number = dense_layers_number
        self.decoded_dropout_rate = decoded_dropout_rate
        self.output_dropout_rate = output_dropout_rate
        self.max_source_length = max_source_length
        self.max_target_lenght = max_target_lenght
        self.vocab_size = vocab_size
        self.feature_columns = feature_columns

        self.source_embeddings = LandmarkEmbeddingV2(embedings_dim=embedding_dims, max_seq_length=max_source_length, kernel_size=encodder_kernel_size)
        self.target_embeddings = BasicPositionalEmbeddings(dim_emb=embedding_dims, max_seq_length=max_target_lenght, max_tokens=vocab_size)
        self.encoder = TransformerEncoder(num_heads=attention_heads, dim_emb=embedding_dims, dim_dense=math.ceil(attention_heads/embedding_dims))
        self.decoder = TransformerDecoder(num_heads=attention_heads, dim_dense=math.ceil(attention_heads/embedding_dims), dim_emb=embedding_dims)

        self.dropout_after_decoder = tf.keras.layers.Dropout(rate=decoded_dropout_rate)
        # keep non zero values
        #Taking hands only x y and just the dominant hand at frame that is why the /2
        self.ffc = tf.keras.Sequential(
                    [tf.keras.layers.Dense(units=int(feature_columns.shape[0]/2), name=f"fccl_{layer_number+1}") for layer_number in range(dense_layers_number)]
                )
        self.dropout_before_output = tf.keras.layers.Dropout(rate=output_dropout_rate)
        self.next_token_proba = tf.keras.layers.Dense(units=vocab_size, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        source_batch = inputs[0]
        target_batch = inputs[1]

        source_embeddings = self.source_embeddings(source_batch)
        encoded_source = self.encoder(source_embeddings)

        target_embeddings = self.target_embeddings(target_batch)
        decoded_sequence = self.decoder(encoded_source, target_embeddings)

        decoded_sequence = self.dropout_after_decoder(decoded_sequence)
        decoded_sequence = self.ffc(decoded_sequence)
        decoded_sequence = self.dropout_before_output(decoded_sequence)

        return self.next_token_proba(decoded_sequence)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, MAX_LENGHT_SOURCE, int(FEATURE_COLUMNS.shape[0]/2)], dtype=tf.float32, name='inputs')])
    def generate(self, source):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        landkaremb = self.source_embeddings(source)
        enc = self.encoder(landkaremb, training = False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * start_token_idx
        dec_logits = []
        for i in range(self.max_target_lenght - 1):
            target_emb = self.target_embeddings(dec_input)
            dec_out = self.decoder(enc, target_emb, training = False)
            logits = self.next_token_proba(self.ffc(dec_out), training=False)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

    def get_config(self):
        config = super().get_config()
        config.update({"attention_heads": self.attention_heads,
                "embedding_dims": self.embedding_dims,
                "encodder_kernel_size": self.encodder_kernel_size,
                "dense_layers_number": self.dense_layers_number,
                "decoded_dropout_rate": self.decoded_dropout_rate,
                "output_dropout_rate": self.output_dropout_rate,
                "max_source_length": self.max_source_length,
                "max_target_lenght": self.max_target_lenght,
                "vocab_size": self.vocab_size,
                "feature_columns": self.feature_columns})
        return config
