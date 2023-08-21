import tensorflow as tf
from src.Transformers.Encoder import TransformerEncoder
from src.Transformers.Decoder import TransformerDecoder
import math
from src.custom.layers import LandmarkEmbedding
from src.Transformers.PositionalEncoding import BasicPositionalEmbeddings


@tf.keras.utils.register_keras_serializable(name="fingerspelling_model_v1")
class FingerSpellingV1(tf.keras.Model):
    def __init__(self, attention_heads, embedding_dims, encodder_kernel_size,
                 dense_layers_number, decoded_output_dropout, output_dropout, max_source_length,
                 max_target_lenght, vocab_size, feature_columns, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.attention_heads = attention_heads
        self.embedding_dims = embedding_dims
        self.encodder_kernel_size = encodder_kernel_size
        self.dense_layers_number = dense_layers_number
        self.decoded_output_dropout = decoded_output_dropout
        self.output_dropout = output_dropout
        self.max_source_length = max_source_length
        self.max_target_lenght = max_target_lenght
        self.latent_dims = math.ceil(attention_heads/embedding_dims)
        self.vocab_size = vocab_size
        self.feature_columns = feature_columns

        self.source_embeddings = LandmarkEmbedding(embedings_dim=embedding_dims, max_seq_length=max_source_length, kernel_size=encodder_kernel_size)

        self.target_embeddings = BasicPositionalEmbeddings(dim_emb=embedding_dims, max_seq_length=max_target_lenght, max_tokens=vocab_size)

        self.encoder = TransformerEncoder(num_heads=attention_heads, dim_emb=embedding_dims, dim_dense=self.latent_dims)

        self.decoder = TransformerDecoder(num_heads=attention_heads, dim_dense=self.latent_dims, dim_emb=embedding_dims)

        self.ffc = tf.keras.Sequential([tf.keras.layers.Dense(units=feature_columns.shape[0], name=f"fccl_{layer_number+1}") for layer_number in range(dense_layers_number)])

        self.decoded_output_dropout = tf.keras.layers.Dropout(rate=decoded_output_dropout)

        self.output_dropout = tf.keras.layers.Dropout(rate=output_dropout)

        self.next_token = tf.keras.layers.Dense(units=vocab_size, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        source_batch = inputs["source"]
        target_batch = inputs["target"]

        source_embeddings = self.source_embeddings(source_batch)
        encoded_source = self.encoder(source_embeddings)

        target_embeddings = self.target_embeddings(target_batch)
        decoded_sequence = self.decoder(encoded_source, target_embeddings)

        decoded_sequence = self.decoded_output_dropout(decoded_sequence)

        ffl = self.ffc(decoded_sequence)
        ffl = self.output_dropout(ffl)

        next_token = self.next_token(ffl)

        return next_token

    def generate_sequence(self, source_sequence_batch, start_token_idx):
        batch_size = tf.shape(source_sequence_batch)[0]
        max_lenght = self.max_target_lenght
        target_sequence = tf.ones((batch_size, 1), dtype=tf.int32) * start_token_idx
        for i in range(max_lenght):
            y_preds = self({"source": source_sequence_batch, "target": target_sequence}, training=False)
            next_tokens = tf.cast(tf.argmax(y_preds, axis=-1), dtype=tf.int32)
            last_tokens = next_tokens[:, -1][..., tf.newaxis]
            target_sequence = tf.concat([target_sequence, last_tokens], axis=-1)

        return preds

    def get_config(self):
        return {"attention_heads": self.attention_heads,
                "embedding_dims": self.embedding_dims,
                "encodder_kernel_size": self.encodder_kernel_size,
                "dense_layers_number": self.dense_layers_number,
                "decoded_output_dropout": self.decoded_output_dropout,
                "output_dropout": self.output_dropout,
                "max_source_length": self.max_source_length,
                "max_target_lenght": self.max_target_lenght,
                "vocab_size": self.vocab_size,
                "feature_columns": self.feature_columns}
