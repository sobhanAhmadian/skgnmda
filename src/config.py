import os

from base.config import Config
from base.config import ModelConfig

RAW_DATA_DIR = os.getcwd() + "/data_repository/raw"
PROCESSED_DATA_DIR = os.getcwd() + "/data_repository/processed"
LOG_DIR = os.getcwd() + "/data_repository/log"
MODEL_SAVED_DIR = os.getcwd() + "/data_repository/ckpt"

INTEGRATED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "Integrated")
VMH_RAW_DATASET_DIRECTOR = os.path.join(RAW_DATA_DIR, "VMH")
MIND_RAW_DATASET_DIRECTOR = os.path.join(RAW_DATA_DIR, "MIND")
HMDAD_RAW_DATASET_DIRECTOR = os.path.join(RAW_DATA_DIR, "HMDAD")

# mdkg_hmdad
KG_FILE = {"mdkg_hmdad": os.path.join(RAW_DATA_DIR, "mdkg_hmdad", "train2id.txt")}
ENTITY2ID_FILE = {
    "mdkg_hmdad": os.path.join(RAW_DATA_DIR, "mdkg_hmdad", "entity2id.txt")
}
RELATION2ID_FILE = {
    "mdkg_hmdad": os.path.join(RAW_DATA_DIR, "mdkg_hmdad", "relation2id.txt")
}
EXAMPLE_FILE = {
    "mdkg_hmdad": os.path.join(RAW_DATA_DIR, "mdkg_hmdad", "approved_example.txt")
}

# add gaussian similarity
MICROBE_SIMILARITY_FILE = os.path.join(PROCESSED_DATA_DIR, "microbesimilarity.csv")
DISEASE_SIMILARITY_FILE = os.path.join(PROCESSED_DATA_DIR, "diseasesimilarity.csv")

SEPARATOR = {"mdkg_hmdad": "\t"}
NEIGHBOR_SIZE = {"mdkg_hmdad": 8}

ENTITY_VOCAB_TEMPLATE = "{dataset}_entity_vocab.pkl"
RELATION_VOCAB_TEMPLATE = "{dataset}_relation_vocab.pkl"
ADJ_ENTITY_TEMPLATE = "{dataset}_adj_entity.npy"
ADJ_RELATION_TEMPLATE = "{dataset}_adj_relation.npy"
TRAIN_DATA_TEMPLATE = "{dataset}_train.npy"
TEST_DATA_TEMPLATE = "{dataset}_test.npy"

RESULT_LOG = {"mdkg_hmdad": "mdkg_result.txt"}
PERFORMANCE_LOG = "MDKGNN_performance.log"
DISEASE_MICROBE_EXAMPLE = "dataset_examples.npy"


class KGCNModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.neighbor_sample_size = None  # neighbor sampling size
        self.n_depth = None  # depth of receptive field
        self.l2_weight = None  # l2 regularize weight
        self.aggregator_type = None

    def get_configuration(self):
        return {
            **super().get_configuration(),
            **{
                "neighbor_sample_size": self.neighbor_sample_size,
                "n_depth": self.n_depth,
                "l2_weight": self.l2_weight,
                "aggregator_type": self.aggregator_type,
            },
        }

    def get_summary(self):
        return {
            **super().get_summary(),
            **{
                "neighbor_sample_size": self.neighbor_sample_size,
                "n_depth": self.n_depth,
                "l2_weight": self.l2_weight,
                "aggregator_type": self.aggregator_type,
            },
        }


class DataConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "mdkg"

        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None

    def get_configuration(self):
        return {
            "dataset": self.dataset,
            "entity_vocab_size": self.entity_vocab_size,
            "relation_vocab_size": self.relation_vocab_size,
        }

    def get_summary(self):
        return {
            "entity_vocab_size": self.entity_vocab_size,
            "relation_vocab_size": self.relation_vocab_size,
        }
