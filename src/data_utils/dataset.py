import tensorflow as tf
import math
import numpy as np
import os
import json
from src.constants import TARGET_MAX_LENGHT, MAX_LENGHT_SOURCE, FEATURES_SIZE, FEATURE_COLUMNS

# Constants
DATASET_DIR = "../data/asl-fingerspelling"
TRAIN_LANDMARKS_PATH = DATASET_DIR+os.sep+"train_landmarks"
SUPPLEMENTAL_LANDMARKS_PATH = DATASET_DIR+os.sep+"supplemental_landmarks"
TRAIN_PATH = DATASET_DIR+os.sep+"train.csv"
TFRECORDS_PATH = DATASET_DIR+os.sep+"tfrecords"

with open ("../data/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num=json.load(f)

# Add pad_token, start pointer and end pointer to the dict
start_token = "<"
end_token = ">"
pad_token = "P"
space_token = " "
unk_token = "[UNK]"
pad_token_idx = 0
space_token_idx = 59
unk_token_idx = -1
start_token_idx = 60
end_token_idx = 61

char_to_num[pad_token] = pad_token_idx
char_to_num[space_token] = space_token_idx
char_to_num[unk_token] = unk_token_idx
char_to_num[start_token] = start_token_idx
char_to_num[end_token] = end_token_idx
num_to_char = {j:i for i,j in char_to_num.items()}
VOCAB_SIZE = len([w for w, _ in num_to_char.items()])

def resize_pad(x, phrase):
    if tf.shape(x)[0] < MAX_LENGHT_SOURCE:
        x = tf.pad(x, ([[0, MAX_LENGHT_SOURCE-tf.shape(x)[0]], [0, 0]]), mode = 'CONSTANT', constant_values=0.0)
    else:
        x = tf.slice(x, [0, 0], [MAX_LENGHT_SOURCE, FEATURES_SIZE])

    return x, phrase

def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks = ([tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLUMNS])
    # Transpose to maintain the original shape of landmarks data.
    landmarks = tf.transpose(landmarks)
    
    return landmarks, phrase

table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=list(char_to_num.keys()),
        values=list(char_to_num.values()),
    ),
    default_value=tf.constant(-1),
    name="class_weight"
)

itext_vectorizer = tf.keras.layers.TextVectorization(
    output_sequence_length=TARGET_MAX_LENGHT,
    output_mode="int",
    standardize=None,
    split="character",
    vocabulary=[w for w, _ in char_to_num.items()]
)

def convert_fn(landmarks, phrase):

    # Add start and end pointers to phrase.
    phrase_with_start_end_token = start_token + phrase + end_token
    phrase_splited = tf.strings.bytes_split(phrase_with_start_end_token)
    phrase_with_indexes = table.lookup(phrase_splited)
    # Vectorize and add padding.
    # 65 added the UNK token
    if tf.shape(phrase_with_indexes)[0] < TARGET_MAX_LENGHT:
        phrase_with_indexes = tf.pad(
                                    phrase_with_indexes,
                                    paddings=[[0, TARGET_MAX_LENGHT-tf.shape(phrase_with_indexes)[0]]],
                                    mode = 'CONSTANT',
                                    constant_values = pad_token_idx
                            )
    else:
        phrase_with_indexes = tf.slice(phrase_with_indexes, [0], [TARGET_MAX_LENGHT])

    return tf.cast(landmarks, dtype=tf.float32), tf.cast(phrase_with_indexes, dtype=tf.int32)


def build_datset_train_val(split=0.8, batch_size=128):

    list_files = np.array([TFRECORDS_PATH+os.sep+"train_landmarks"+os.sep+file_name for file_name in os.listdir(TFRECORDS_PATH+os.sep+"train_landmarks/")])

    N = list_files.shape[0]*512

    train_split = math.ceil(N*split)
    validation_split = N - train_split

    number_of_train_files = math.ceil(train_split/512)

    np.random.shuffle(list_files)

    train_files = list_files[:number_of_train_files]
    val_files = list_files[number_of_train_files:]

    print(f"train split: {train_files.shape[0]*512} | val split: {val_files.shape[0]*512}")

    train_dataset = (tf.data.TFRecordDataset(train_files)
                        .shuffle(batch_size)
                        .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(resize_pad, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(convert_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda landmark, phrase: ({"source":landmark, "target":phrase[:-1]}, phrase[1:]))
                        .batch(batch_size)
                        .prefetch(tf.data.AUTOTUNE)
                    )

    val_dataset = (tf.data.TFRecordDataset(val_files)
                        .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(resize_pad, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(convert_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda landmark, phrase: ({"source":landmark, "target":phrase[:-1]}, phrase[1:]))
                        .cache()
                        .batch(batch_size)
                        .prefetch(tf.data.AUTOTUNE)
                    )

    return train_dataset, val_dataset
