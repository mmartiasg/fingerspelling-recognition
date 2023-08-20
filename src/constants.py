import math
import numpy as np
import json

MAX_LENGHT_SOURCE = 120
DIM_EMBEDDINGS = 512
ATTENTION_HEADS = 8

with open ("../data/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num=json.load(f)

LATENT_DIMS = math.ceil(DIM_EMBEDDINGS/ATTENTION_HEADS)
FEATURE_COLUMNS = np.load("../data/asl-fingerspelling/feature_columns.npy")
FEATURES_SIZE = FEATURE_COLUMNS.shape[0]
TARGET_MAX_LENGHT = 64
