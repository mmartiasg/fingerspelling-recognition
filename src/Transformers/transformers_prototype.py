from src.constants import MAX_LENGHT_SOURCE, FEATURES_SIZE, LATENT_DIMS, DIM_EMBEDDINGS, TARGET_MAX_LENGHT, ATTENTION_HEADS
from src.data_utils.dataset import VOCAB_SIZE
import tensorflow as tf
from src.custom.layers import LandmarkEmbedding, LandmarkEmbeddingV2
from src.Transformers.Encoder import TransformerEncoder
from src.Transformers.Decoder import TransformerDecoder
from src.Transformers.PositionalEncoding import BasicPositionalEmbeddings
import math
from src.custom.metrics import SparseLevenshtein, SparseLevenshteinV2


def build_transformer_model_v1(trial):

    ATTENTION_HEADS = 1
    LATENT_DIMS = 64
    LR = 1e-3
    DROPOUT_RATE_DECODER=0.1
    DROPOUT_RATE_OUTPUT=0.2
    DENSE_LAYERS = 1
    ENCODER_KERNEL_SIZE = 7

    if trial is not None:
        ATTENTION_HEADS = trial.suggest_int('attention_heads', 1, 8)
        LATENT_DIMS = math.ceil(DIM_EMBEDDINGS/ATTENTION_HEADS)
        LR = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
        DROPOUT_RATE_DECODER = trial.suggest_float("drop_out_out_decoder", 0.0, 0.5, step=0.05)
        DENSE_LAYERS = trial.suggest_int('dense_layers', 1, 5)
        DROPOUT_RATE_OUTPUT = trial.suggest_float("drop_out", 0.0, 0.5, step=0.05)
        ENCODER_KERNEL_SIZE = trial.suggest_int('encoder_kernel_size', 3, 12)

    # INPUT SOURCE
    source = tf.keras.layers.Input(shape=(MAX_LENGHT_SOURCE, FEATURES_SIZE), dtype=tf.float32, name="source")
    source_emb = LandmarkEmbedding(embedings_dim=DIM_EMBEDDINGS, max_seq_length=MAX_LENGHT_SOURCE, kernel_size=ENCODER_KERNEL_SIZE)(source)

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
    decoded_target_sequence = tf.keras.layers.Dropout(rate=DROPOUT_RATE_DECODER)(decoded_target_sequence)
    # DECODER END

    for layer_index in range(DENSE_LAYERS):
        decoded_target_sequence = tf.keras.layers.Dense(units=FEATURES_SIZE, name=f"dense_layers_{layer_index}")(decoded_target_sequence)

    decoded_target_sequence = tf.keras.layers.Dropout(rate=DROPOUT_RATE_OUTPUT)(decoded_target_sequence)

    next_token = tf.keras.layers.Dense(units=VOCAB_SIZE, activation="softmax")(decoded_target_sequence)

    # CREATE MODEL ENCODER / DECODER
    encoder_decoder_model = tf.keras.Model(inputs=[source, target], outputs=next_token)

    encoder_decoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 metrics=["accuracy", SparseLevenshtein()],
                                 #XLA compilation issues with device location of int32 variables cannot be placed in GPU
                                 jit_compile=False
    )

    return encoder_decoder_model


def build_transformer_model_v2(trial):

    ATTENTION_HEADS = 1
    LATENT_DIMS = 64
    LR = 1e-3
    DROPOUT_RATE_DECODER=0.1
    DROPOUT_RATE_OUTPUT=0.2
    DENSE_LAYERS = 1
    ENCODER_KERNEL_SIZE = 7

    if trial is not None:
        ATTENTION_HEADS = trial.suggest_int('attention_heads', 1, 8)
        LATENT_DIMS = math.ceil(DIM_EMBEDDINGS/ATTENTION_HEADS)
        LR = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        DROPOUT_RATE_DECODER = trial.suggest_float("drop_out_out_decoder", 0.0, 0.5, step=0.05)
        DENSE_LAYERS = trial.suggest_int('dense_layers', 1, 5)
        DROPOUT_RATE_OUTPUT = trial.suggest_float("drop_out", 0.0, 0.5, step=0.05)
        ENCODER_KERNEL_SIZE = trial.suggest_int('encoder_kernel_size', 3, 12)

    # INPUT SOURCE
    source = tf.keras.layers.Input(shape=(MAX_LENGHT_SOURCE, FEATURES_SIZE), dtype=tf.float32, name="source")
    source_emb = LandmarkEmbeddingV2(embedings_dim=DIM_EMBEDDINGS, max_seq_length=MAX_LENGHT_SOURCE, kernel_size=ENCODER_KERNEL_SIZE)(source)

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
    decoded_target_sequence = tf.keras.layers.Dropout(rate=DROPOUT_RATE_DECODER)(decoded_target_sequence)
    # DECODER END

    for layer_index in range(DENSE_LAYERS):
        decoded_target_sequence = tf.keras.layers.Dense(units=FEATURES_SIZE, name=f"dense_layers_{layer_index}")(decoded_target_sequence)

    decoded_target_sequence = tf.keras.layers.Dropout(rate=DROPOUT_RATE_OUTPUT)(decoded_target_sequence)

    next_token = tf.keras.layers.Dense(units=VOCAB_SIZE, activation="softmax")(decoded_target_sequence)

    # CREATE MODEL ENCODER / DECODER
    encoder_decoder_model = tf.keras.Model(inputs=[source, target], outputs=next_token)

    #XLA compilation has issues with device location of int32 variables cannot be placed in GPU
    encoder_decoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 metrics=["accuracy", SparseLevenshtein()]
    )

    return encoder_decoder_model
