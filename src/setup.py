import os

import numpy as np

from src import config
from src.data.microbe_disease_data import process_data


def load_similarity(similarity_file):
    term_id_similarity = {}
    with open(similarity_file, "r") as reader:
        for line in reader:
            term_id, term_similarity = line.strip().split(":")
            term_similarity_array = np.array(term_similarity.strip().split("\t"))
            term_id_similarity[int(term_id)] = term_similarity_array
    return term_id_similarity, len(term_similarity_array)


def generate_pre_embedding(matrix_row, matrix_column, data_dict):
    pre_embed = np.zeros((matrix_row, matrix_column), dtype="float64")
    for key1 in data_dict:
        for i in range(len(data_dict[key1])):
            pre_embed[key1][i] = data_dict[key1][i]
    return pre_embed


if __name__ == "__main__":
    if not os.path.exists(config.PROCESSED_DATA_DIR):
        os.makedirs(config.PROCESSED_DATA_DIR)
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    if not os.path.exists(config.MODEL_SAVED_DIR):
        os.makedirs(config.MODEL_SAVED_DIR)
    model_config = config.ModelConfig()
    process_data("mdkg_hmdad", config.NEIGHBOR_SIZE["mdkg_hmdad"], 5)
