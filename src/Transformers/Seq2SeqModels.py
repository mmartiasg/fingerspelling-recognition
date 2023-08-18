import tensorflow as tf
from Transformers.Encoder import TransformerEncoder
from Transformers.Decoder import TransformerDecoder
from Transformers.PositionalEncoding import BasicPositionalEmbeddings


class TransformerBasedTranslator(tf.keras.Model):
    def __init__(self, num_heads, dim_embeddings, dim_dense, source_max_tokens, target_max_tokens,
                 source_max_seq_length, target_max_seq_length, dropout_rate=0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_heads = num_heads
        self.dim_embeddings = dim_embeddings
        self.dim_dense = dim_dense
        self.source_max_tokens = source_max_tokens
        self.target_max_tokens = target_max_tokens
        self.source_max_seq_length = source_max_seq_length
        self.target_max_seq_length = target_max_seq_length
        self.dropout_rate = dropout_rate
        self.source_positional_encoding = BasicPositionalEmbeddings(dim_emb=dim_embeddings,
                                                                    max_tokens=source_max_tokens,
                                                                    max_seq_length=source_max_seq_length)
        self.target_positional_encoding = BasicPositionalEmbeddings(dim_emb=dim_embeddings,
                                                                    max_tokens=target_max_tokens,
                                                                    max_seq_length=target_max_seq_length)
        self.encoder = TransformerEncoder(num_heads=num_heads, dim_dense=dim_dense, dim_emb=dim_embeddings)
        self.decoder = TransformerDecoder(num_heads=num_heads, dim_dense=dim_dense, dim_emb=dim_embeddings)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.next_token = tf.keras.layers.Dense(units=target_max_tokens, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        source_embeddings = self.source_positional_encoding(inputs["source"])
        encoded_seq = self.encoder(source_embeddings)
        target_embeddings = self.target_positional_encoding(inputs["target"])
        decoded_seq = self.decoder(encoded_seq, target_embeddings)
        drop_decoded_seq = self.dropout(decoded_seq)

        return self.next_token(drop_decoded_seq)

    def get_config(self):
        return {"num_heads": self.num_heads,
                "dim_embeddings": self.dim_embeddings,
                "dim_dense": self.dim_dense,
                "source_max_tokens": self.source_max_tokens,
                "target_max_tokens": self.target_max_tokens,
                "source_max_seq_length": self.source_max_seq_length,
                "target_max_seq_length": self.target_max_seq_length,
                "dropout_rate": self.dropout_rate}
