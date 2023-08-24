import tensorflow as tf
from src.constants import MAX_LENGHT_SOURCE, FEATURES_SIZE, LATENT_DIMS, DIM_EMBEDDINGS, TARGET_MAX_LENGHT, ATTENTION_HEADS
from src.data_utils.dataset import VOCAB_SIZE, FEATURE_COLUMNS
import math
from src.custom.metrics import SparseLevenshtein
from src.prod_models.Fingerspelling import FingerSpellingV1, FingerSpellingV2

def build_prod_transformer_model_v1(trial):
    ATTENTION_HEADS = 1
    LR = 1e-3
    DROPOUT_RATE_DECODER=0.1
    DROPOUT_RATE_OUTPUT=0.2
    DENSE_LAYERS = 1
    ENCODER_KERNEL_SIZE = 7

    if trial is not None:
        ATTENTION_HEADS = trial.suggest_int('attention_heads', 1, 8)
        LR = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        DROPOUT_RATE_DECODER = trial.suggest_float("drop_out_out_decoder", 0.0, 0.5, step=0.05)
        DENSE_LAYERS = trial.suggest_int('dense_layers', 1, 5)
        DROPOUT_RATE_OUTPUT = trial.suggest_float("drop_out", 0.0, 0.5, step=0.05)
        ENCODER_KERNEL_SIZE = trial.suggest_int('encoder_kernel_size', 3, 12)

    encoder_decoder_model = FingerSpellingV1(attention_heads=ATTENTION_HEADS,
                                                embedding_dims=DIM_EMBEDDINGS,
                                                encodder_kernel_size=ENCODER_KERNEL_SIZE,
                                                dense_layers_number=DENSE_LAYERS,
                                                decoded_dropout_rate=DROPOUT_RATE_DECODER,
                                                output_dropout_rate=DROPOUT_RATE_OUTPUT,
                                                max_source_length=MAX_LENGHT_SOURCE,
                                                max_target_lenght=TARGET_MAX_LENGHT,
                                                vocab_size=VOCAB_SIZE,
                                                feature_columns=FEATURE_COLUMNS
                            )

    #XLA compilation has issues with device location of int32 variables cannot be placed in GPU
    encoder_decoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 metrics=["accuracy", SparseLevenshtein()]
    )

    return encoder_decoder_model

def build_prod_transformer_model_v2(trial):
    ATTENTION_HEADS = 12
    LR = 1e-4
    DROPOUT_RATE_DECODER=0.1
    DROPOUT_RATE_OUTPUT=0.2
    DENSE_LAYERS = 5
    ENCODER_KERNEL_SIZE = 10

    if trial is not None:
        # Second version change this to 10
        ATTENTION_HEADS = trial.suggest_int('attention_heads', 1, 8)
        LR = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        DROPOUT_RATE_DECODER = trial.suggest_float("drop_out_out_decoder", 0.0, 0.5, step=0.05)
        DENSE_LAYERS = trial.suggest_int('dense_layers', 1, 5)
        DROPOUT_RATE_OUTPUT = trial.suggest_float("drop_out", 0.0, 0.5, step=0.05)
        ENCODER_KERNEL_SIZE = trial.suggest_int('encoder_kernel_size', 3, 12)

    encoder_decoder_model = FingerSpellingV2(attention_heads=ATTENTION_HEADS,
                                                embedding_dims=DIM_EMBEDDINGS,
                                                encodder_kernel_size=ENCODER_KERNEL_SIZE,
                                                dense_layers_number=DENSE_LAYERS,
                                                decoded_dropout_rate=DROPOUT_RATE_DECODER,
                                                output_dropout_rate=DROPOUT_RATE_OUTPUT,
                                                max_source_length=MAX_LENGHT_SOURCE,
                                                max_target_lenght=TARGET_MAX_LENGHT,
                                                vocab_size=VOCAB_SIZE,
                                                feature_columns=FEATURE_COLUMNS
                            )

    #XLA compilation has issues with device location of int32 variables cannot be placed in GPU
    encoder_decoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 metrics=["accuracy", SparseLevenshtein()]
    )

    return encoder_decoder_model
