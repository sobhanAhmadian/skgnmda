import sklearn.metrics as m
from keras.callbacks import Callback
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
)

from base.data import Data


class KGCNMetric(Callback):
    def __init__(self, data: Data, aggregator_type, dataset, K_fold, log_file_path):
        self.x_train = data.X
        self.y_train = data.y
        self.aggregator_type = aggregator_type
        self.dataset = dataset
        self.k = K_fold
        self.threshold = 0.5
        self.log_file_path = log_file_path
        super(KGCNMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        y_predicted = self.model.predict(self.x_train).flatten()
        y_true = self.y_train.flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_predicted)  # roc曲线的auc
        precision, recall, _thresholds = precision_recall_curve(
            y_true=y_true, probas_pred=y_predicted
        )
        aupr = m.auc(recall, precision)
        y_predicted = [1 if prob >= self.threshold else 0 for prob in y_predicted]
        acc = accuracy_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)

        logs["train_aupr"] = float(aupr)
        logs["train_auc"] = float(auc)
        logs["train_acc"] = float(acc)
        logs["train_f1"] = float(f1)

        logs["dataset"] = self.dataset
        logs["aggregator_type"] = self.aggregator_type
        logs["k_fold"] = self.k
        logs["epoch_count"] = epoch + 1
        print(
            f"Logging Info - epoch: {epoch + 1}, train_auc: {auc}, train_aupr: {aupr}, train_acc: {acc}, train_f1: {f1}"
        )
