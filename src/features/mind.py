from src.config import MIND_RAW_DATASET_DIRECTOR
import os
from src.utils import json_load

NETWORKS = {
    "C0310": os.path.join(MIND_RAW_DATASET_DIRECTOR, "C0310.json"),
}


def get_raw_network(name):
    if name not in NETWORKS.keys():
        raise Exception(f"No network with name {name}!")
    return json_load(NETWORKS[name])

