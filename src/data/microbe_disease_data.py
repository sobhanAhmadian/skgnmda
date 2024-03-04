import numpy as np

from base.data import Data, TrainTestSplit
from src.methods.similarity_metrics import get_gaussian_similarity


class MicrobeDiseaseData(Data):
    def extend(self, X, y):
        if self.X is None or self.y is None:
            self.X = X
            self.y = y
        else:
            self.X[0] = np.append(self.X[0], X[0], axis=0)
            self.X[1] = np.append(self.X[1], X[1], axis=0)
            self.y = np.append(self.y, y, axis=0)

    def subset(self, indices) -> tuple:
        return [self.X[0][indices, :], self.X[1][indices, :]], self.y[indices]


class MicrobeDiseaseTrainTestSplit(TrainTestSplit):
    def __init__(self, examples, with_gaussian_similarity=False):
        self.examples = examples
        self.with_gaussian_similarity = with_gaussian_similarity

    def split(self, train_indices, test_indices):
        train_diseases = self.examples[train_indices, :1]
        train_microbes = self.examples[train_indices, 1:2]
        train_associations = self.examples[train_indices, 2:3].reshape(-1)

        test_diseases = self.examples[test_indices, :1]
        test_microbes = self.examples[test_indices, 1:2]
        test_associations = self.examples[test_indices, 2:3].reshape(-1)
        test_examples = self.examples[test_indices, :]

        train_data = MicrobeDiseaseData(
            [train_diseases, train_microbes], train_associations
        )
        test_data = MicrobeDiseaseData(
            [test_diseases, test_microbes], test_associations
        )

        if self.with_gaussian_similarity:
            (
                train_disease_similarities,
                train_microbe_similarities,
                test_disease_similarities,
                test_microbe_similarities,
            ) = get_gaussian_similarity(self.examples, test_examples)
            train_data = MicrobeDiseaseData(
                [
                    train_diseases,
                    train_microbes,
                    train_disease_similarities,
                    train_microbe_similarities,
                ],
                train_associations,
            )
            test_data = MicrobeDiseaseData(
                [
                    test_diseases,
                    test_microbes,
                    test_disease_similarities,
                    test_microbe_similarities,
                ],
                test_associations,
            )
        return train_data, test_data
