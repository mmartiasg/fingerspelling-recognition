from src.constants import MAX_LENGHT_SOURCE, FEATURES_SIZE, LATENT_DIMS, DIM_EMBEDDINGS, TARGET_MAX_LENGHT
from src.data_utils.dataset import VOCAB_SIZE
import tensorflow as tf


def build_gru_model(trial):
    #input
    source = tf.keras.layers.Input(shape=(MAX_LENGHT_SOURCE, FEATURES_SIZE), dtype="float32", name="source")
    # source_emb = LandmarkEmbedding(embedings_dim=DIM_EMBEDDINGS)(source)

    # ENCODER START
    encoded_source = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=LATENT_DIMS, return_sequences=True, recurrent_dropout=0.25), merge_mode="sum")(source)
    for layer_index in range(trial.suggest_int('encoder_rnn_layers', 1, 7, step=1)):
        index_correction = layer_index+2
        encoded_source = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=LATENT_DIMS,
                                                                            return_sequences=True,
                                                                            recurrent_dropout=0.25),
                                                                            merge_mode="sum",
                                                                            name=f"encoder_gru_{index_correction}")(encoded_source)

    encoded_source = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=LATENT_DIMS,
                                                                    return_sequences=False,
                                                                    recurrent_dropout=0.25),
                                                                    merge_mode="sum",
                                                                    name=f"encoder_gru_{index_correction+1}")(encoded_source)
    # ENCODER END

    # DECODER
    # Encodded token and new generated token as inputs
    # Reverse process like the conv1DTranspose with the segmentation model
    target = tf.keras.Input(shape=(TARGET_MAX_LENGHT+1,), dtype="int32", name="target")

    #learn a representation from the target (Vocabsize, latent_dim)
    #Should this be MAX_TOKENS+1 for the END?
    latent_space = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=DIM_EMBEDDINGS, mask_zero=True)(target)

    #This will return (TARGET_SHAPE, LATENT_DIM)
    decoder = tf.keras.layers.GRU(units=LATENT_DIMS, return_sequences=True)
    decoded_sentence = decoder(latent_space, initial_state=encoded_source)
    decoded_sentence = tf.keras.layers.Dropout(rate=trial.suggest_float("dropout", 0.0, 0.5))(decoded_sentence)

    target_next_step = tf.keras.layers.Dense(units=VOCAB_SIZE, activation="softmax")(decoded_sentence)
    # DECODER

    # CREATE MODEL ENCODER / DECODER
    encoder_decoder_model = tf.keras.Model(inputs=[source, target], outputs=target_next_step)

    encoder_decoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"]
    )

    return encoder_decoder_model
