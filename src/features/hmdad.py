from src.utils import json_load
import os
from src.config import HMDAD_RAW_DATASET_DIRECTOR


def _get_table(table_name):
    return json_load(os.path.join(HMDAD_RAW_DATASET_DIRECTOR, f"{table_name}.json"))


def get_raw_microbe_diseases():
    association_data = _get_table("microbe_disease")
    return association_data
