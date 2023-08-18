from src.constants import MAX_LENGHT_SOURCE, FEATURES_SIZE, LATENT_DIMS, DIM_EMBEDDINGS, TARGET_MAX_LENGHT, ATTENTION_HEADS
from src.data_utils.dataset import VOCAB_SIZE
import tensorflow as tf
import tensorflow_models as tfm
from src.custom.layers import LandmarkEmbedding
from src.Transformers.Encoder import TransformerEncoder
from src.Transformers.Decoder import TransformerDecoder
from src.Transformers.PositionalEncoding import BasicPositionalEmbeddings
import math


def build_transformer_model(trial):

    ATTENTION_HEADS = trial.suggest_int('attention_heads', 1, 8, step=2)
    LATENT_DIMS = math.ceil(DIM_EMBEDDINGS/ATTENTION_HEADS)

    # INPUT SOURCE
    source = tf.keras.layers.Input(shape=(MAX_LENGHT_SOURCE, FEATURES_SIZE), dtype=tf.float32, name="source")
    source_emb = LandmarkEmbedding(embedings_dim=DIM_EMBEDDINGS, max_seq_length=MAX_LENGHT_SOURCE)(source)

    # ENCODER START
    # Auto encoder
    encoder = TransformerEncoder(num_heads=ATTENTION_HEADS, dim_emb=DIM_EMBEDDINGS, dim_dense=LATENT_DIMS)
    encoded_source_sequence = encoder(source_emb)    
    # ENCODER END

    # INPUT TARGET
    target = tf.keras.layers.Input(shape=(TARGET_MAX_LENGHT,), dtype=tf.int32, name="target")
    
    # DECODER START
    target_emb = BasicPositionalEmbeddings(max_tokens=VOCAB_SIZE, max_seq_length=TARGET_MAX_LENGHT+1, dim_emb=DIM_EMBEDDINGS)(target)
    decoder = TransformerDecoder(num_heads=ATTENTION_HEADS, dim_emb=DIM_EMBEDDINGS, dim_dense=LATENT_DIMS)

    decoded_target_sequence = decoder(encoded_source_sequence, target_emb)
    decoded_target_sequence = tf.keras.layers.Dropout(rate=trial.suggest_float("drop_out_out_decoder", 0.0, 0.5))(decoded_target_sequence)
    # DECODER END

    for layer_index in range(trial.suggest_int('dense_layers', 1, 5, step=1)):
        decoded_target_sequence = tf.keras.layers.Dense(units=FEATURES_SIZE, name=f"dense_layers_{layer_index}")(decoded_target_sequence)

    decoded_target_sequence = tf.keras.layers.Dropout(rate=trial.suggest_float("drop_out", 0.0, 0.5))(decoded_target_sequence)

    next_token = tf.keras.layers.Dense(units=VOCAB_SIZE, activation="softmax")(decoded_target_sequence)

    # CREATE MODEL ENCODER / DECODER
    encoder_decoder_model = tf.keras.Model(inputs=[source, target], outputs=next_token)

    encoder_decoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"],
                                 jit_compile=True
    )

    return encoder_decoder_model
