import random
from collections import defaultdict

import numpy as np

from src import config
from src.utils import pickle_dump, format_filename


def read_id_file(file_path: str):
    print(f"Logging Info - Reading id file: {file_path}")
    vocab = {}
    with open(file_path, encoding="utf8") as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            _, entity = line.strip().split("\t")
            vocab[entity] = len(vocab)  # entity_vocab:{'0':0,...}
    return vocab


def read_example_file(file_path: str, separator: str, entity_vocab: dict):
    print(f"Logging Info - Reading example file: {file_path}")
    assert len(entity_vocab) > 0
    examples = []
    with open(file_path, encoding="utf8") as reader:
        for idx, line in enumerate(reader):
            d1, d2, flag = line.strip().split(separator)[:3]
            if d1 not in entity_vocab or d2 not in entity_vocab:
                continue
            if d1 in entity_vocab and d2 in entity_vocab:
                examples.append([entity_vocab[d1], entity_vocab[d2], int(flag)])

    examples_matrix = np.array(examples)
    print(f"size of example: {examples_matrix.shape}")

    return examples_matrix


def read_kg(
    file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int
):
    print(f"Logging Info - Reading kg file: {file_path}")

    kg = defaultdict(list)
    with open(file_path, encoding="utf8") as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            # head, tail, relation = line.strip().split(' ')
            head, relation, tail = line.strip().split("\t")
            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append(
                (entity_vocab[tail], relation_vocab[relation])
            )
            kg[entity_vocab[tail]].append(
                (entity_vocab[head], relation_vocab[relation])
            )

    print(
        f"Logging Info - num of entities: {len(entity_vocab)}, "
        f"num of relations: {len(relation_vocab)}"
    )

    print("Logging Info - Constructing adjacency matrix...")
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    random.seed(1)
    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        if n_neighbor > 0:
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True,
            )

            adj_entity[entity_id] = np.array(
                [all_neighbors[i][0] for i in sample_indices]
            )
            adj_relation[entity_id] = np.array(
                [all_neighbors[i][1] for i in sample_indices]
            )
    print("\n\nadj_entity first 10=\n", adj_entity[:10,])
    print("\n\nadj_relation first 10=\n", adj_relation[:10,])
    return adj_entity, adj_relation


def process_data(dataset: str, neighbor_sample_size: int):
    entity_vocab = read_id_file(config.ENTITY2ID_FILE[dataset])
    relation_vocab = read_id_file(config.RELATION2ID_FILE[dataset])

    examples_file = format_filename(
        config.PROCESSED_DATA_DIR, config.DISEASE_MICROBE_EXAMPLE, dataset=dataset
    )
    examples = read_example_file(
        config.EXAMPLE_FILE[dataset], config.SEPARATOR[dataset], entity_vocab
    )
    np.save(examples_file, examples)  # save examples

    adj_entity, adj_relation = read_kg(
        config.KG_FILE[dataset], entity_vocab, relation_vocab, neighbor_sample_size
    )
    pickle_dump(
        format_filename(
            config.PROCESSED_DATA_DIR, config.ENTITY_VOCAB_TEMPLATE, dataset=dataset
        ),
        entity_vocab,
    )  # save entity_vocab
    pickle_dump(
        format_filename(
            config.PROCESSED_DATA_DIR, config.RELATION_VOCAB_TEMPLATE, dataset=dataset
        ),
        relation_vocab,
    )  # save relation_vocab
    adj_entity_file = format_filename(
        config.PROCESSED_DATA_DIR, config.ADJ_ENTITY_TEMPLATE, dataset=dataset
    )
    np.save(adj_entity_file, adj_entity)  # save adj_entity
    print("Logging Info - Saved:", adj_entity_file)
    adj_relation_file = format_filename(
        config.PROCESSED_DATA_DIR, config.ADJ_RELATION_TEMPLATE, dataset=dataset
    )
    np.save(adj_relation_file, adj_relation)  # save adj_relation
    print("Logging Info - Saved:", adj_relation_file)
