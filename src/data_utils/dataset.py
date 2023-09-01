import tensorflow as tf
import math
import numpy as np
import os
import json
from src.constants import TARGET_MAX_LENGHT, MAX_LENGHT_SOURCE

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
unk_token = "[UNK]"
pad_token_idx = 59
unk_token_idx = -1
start_token_idx = 60
end_token_idx = 61

char_to_num[pad_token] = pad_token_idx
char_to_num[unk_token] = unk_token_idx
char_to_num[start_token] = start_token_idx
char_to_num[end_token] = end_token_idx
num_to_char = {j:i for i,j in char_to_num.items()}

# remove padding token
VOCAB_SIZE = len([w for w, _ in num_to_char.items()]) - 1


LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
# Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

# FEATURE_COLUMNS = X + Y + Z
FEATURE_COLUMNS = np.array(X + Y)

X_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
Y_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
# Z_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

RHAND_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
LHAND_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]

RPOSE_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [int(i) for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]


# Function to resize and add padding.
def resize_pad(x):
    if tf.shape(x)[0] < MAX_LENGHT_SOURCE:
        x = tf.pad(x, ([[0, MAX_LENGHT_SOURCE-tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (MAX_LENGHT_SOURCE, tf.shape(x)[1]))
    return x


def pre_process(x):
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)
    
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)
    
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)
    
    # For dominant hand
    if rnans > lnans:
        hand = lhand
        pose = lpose
        
        hand_x = hand[:, 0*(len(LHAND_IDX)//2) : 1*(len(LHAND_IDX)//2)]
        hand_y = hand[:, 1*(len(LHAND_IDX)//2) : 2*(len(LHAND_IDX)//2)]
        # hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
        # hand = tf.concat([1-hand_x, hand_y, hand_z], axis=1)

        hand = tf.concat([1-hand_x, hand_y], axis=1)
        
        pose_x = pose[:, 0*(len(LPOSE_IDX)//2) : 1*(len(LPOSE_IDX)//2)]
        pose_y = pose[:, 1*(len(LPOSE_IDX)//2) : 2*(len(LPOSE_IDX)//2)]
        # pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
        # pose = tf.concat([1-pose_x, pose_y, pose_z], axis=1)
        pose = tf.concat([1-pose_x, pose_y], axis=1)

    else:
        hand = rhand
        pose = rpose
    
    hand_x = hand[:, 0*(len(LHAND_IDX)//2) : 1*(len(LHAND_IDX)//2)]
    hand_y = hand[:, 1*(len(LHAND_IDX)//2) : 2*(len(LHAND_IDX)//2)]
    # hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
    # hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis]], axis=-1)
    
    # mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    # std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    # hand = (hand - mean) / std

    pose_x = pose[:, 0*(len(LPOSE_IDX)//2) : 1*(len(LPOSE_IDX)//2)]
    pose_y = pose[:, 1*(len(LPOSE_IDX)//2) : 2*(len(LPOSE_IDX)//2)]
    # pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
    # pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)
    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis]], axis=-1)
    
    x = tf.concat([hand, pose], axis=1)
    # x=hand
    x = resize_pad(x)
    
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    x = tf.reshape(x, (MAX_LENGHT_SOURCE, len(LHAND_IDX) + len(LPOSE_IDX)))

    # not taking the z coordinates
    # x = tf.reshape(x, (MAX_LENGHT_SOURCE, len(LHAND_IDX)))
    return x


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

def convert_fn(landmarks, phrase):
    # Add start and end pointers to phrase.
    phrase = start_token + phrase + end_token
    phrase = tf.strings.bytes_split(phrase)
    phrase = table.lookup(phrase)
    # Vectorize and add padding.
    phrase = tf.pad(phrase,
                    paddings=[[0, TARGET_MAX_LENGHT - tf.shape(phrase)[0] + 1]],
                    mode = 'CONSTANT',
                    constant_values = pad_token_idx
            )
    # Apply pre_process function to the landmarks.
    return pre_process(landmarks), tf.cast(phrase, dtype=tf.int32)

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
                        .map(convert_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        # .map(lambda landmark, phrase: ({"source":landmark, "target":phrase[:-1]}, phrase[1:]), num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda landmark, phrase: ((landmark, phrase[:-1]), phrase[1:]), num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(batch_size)
                        .prefetch(tf.data.AUTOTUNE)
                    )

    val_dataset = (tf.data.TFRecordDataset(val_files)
                        .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .map(convert_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        # .map(lambda landmark, phrase: ({"source":landmark, "target":phrase[:-1]}, phrase[1:]), num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda landmark, phrase: ((landmark, phrase[:-1]), phrase[1:]), num_parallel_calls=tf.data.AUTOTUNE)
                        .cache()
                        .batch(batch_size)
                        .prefetch(tf.data.AUTOTUNE)
                    )

    return train_dataset, val_dataset
