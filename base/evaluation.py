from base.data import Data


class Result:
    def __init__(self) -> None:
        self.auc = 0
        self.acc = 0
        self.f1 = 0
        self.aupr = 0

    def get_result(self):
        return {
            "AUC": self.auc,
            "ACC": self.acc,
            "F1 Score": self.f1,
            "AUPR": self.aupr,
        }

    def add(self, result):
        self.auc = self.auc + result.auc
        self.acc = self.acc + result.acc
        self.f1 = self.f1 + result.f1
        self.aupr = self.aupr + result.aupr

    def divide(self, k):
        self.auc = self.auc / k
        self.acc = self.acc / k
        self.f1 = self.f1 / k
        self.aupr = self.aupr / k


def evaluate(data: Data):
    pass
