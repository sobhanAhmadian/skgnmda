import os
import time

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

from base.config import OptimizerConfig
from base.data import Data
from base.evaluation import Result
from base.optimization import Tester, Trainer
from src.optimization.callbacks.ensemble import SWA
from src.models.graph_models import PairKGCN
from keras.losses import BinaryCrossentropy
import tensorflow as tf


class KGCNTrainer(Trainer):
    def train(self, model: PairKGCN, data: Data,
              optimizer_config: OptimizerConfig, metrics) -> Result:
        callbacks = self.add_callbacks(model, optimizer_config)
        for metric in metrics:
            callbacks.append(metric)

        start_time = time.time()
        model.model.compile(
            optimizer=get_optimizer(optimizer_config.optimizer, optimizer_config.lr),
            loss=BinaryCrossentropy(from_logits=True),
            metrics=['acc', 'mae', tf.keras.metrics.AUC()])
        model.model.fit(x=data.X,
                        y=data.y,
                        batch_size=optimizer_config.batch_size,
                        epochs=optimizer_config.n_epoch,
                        callbacks=callbacks)
        elapsed_time = time.time() - start_time
        print(
            "Logging Info - Training time: %s"
            % time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        )

        # if 'swa' in optimizer_config.callbacks_to_add:
        #     model.load_weights(os.path.join(optimizer_config.checkpoint_dir,
        #                                     f'{optimizer_config.exp_name}_swa.hdf5'))
        # else:
        #     model.load_weights(os.path.join(optimizer_config.checkpoint_dir,
        #                                     f'{optimizer_config.exp_name}.hdf5'))

        result = model.evaluate(data)
        return result

    @staticmethod
    def add_callbacks(model, optimizer_config):
        callbacks = []
        if 'modelcheckpoint' in optimizer_config.callbacks_to_add:
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(optimizer_config.checkpoint_dir,
                                      '{}.hdf5'.format(optimizer_config.exp_name)),
                monitor=optimizer_config.checkpoint_monitor,
                save_best_only=optimizer_config.checkpoint_save_best_only,
                save_weights_only=optimizer_config.checkpoint_save_weights_only,
                mode=optimizer_config.checkpoint_save_weights_mode,
                verbose=optimizer_config.checkpoint_verbose
            ))
            print('Logging Info - Callback Added: ModelCheckPoint...')
        if 'earlystopping' in optimizer_config.callbacks_to_add:
            callbacks.append(EarlyStopping(
                monitor=optimizer_config.early_stopping_monitor,
                mode=optimizer_config.early_stopping_mode,
                patience=optimizer_config.early_stopping_patience,
                verbose=optimizer_config.early_stopping_verbose
            ))
        if 'swa' in optimizer_config.callbacks_to_add:
            callbacks.append(SWA(model.build(),
                                 optimizer_config.checkpoint_dir,
                                 optimizer_config.exp_name,
                                 optimizer_config.swa_start))
        return callbacks


class KGCNTester(Tester):
    def test(self, model: PairKGCN, data: Data) -> Result:
        result = model.evaluate(data)
        return result


def get_optimizer(op_type, learning_rate):
    if op_type == "sgd":
        return optimizers.SGD(learning_rate)
    elif op_type == "rmsprop":
        return optimizers.RMSprop(learning_rate)
    elif op_type == "adagrad":
        return optimizers.Adagrad(learning_rate)
    elif op_type == "adadelta":
        return optimizers.Adadelta(learning_rate)
    elif op_type == "adam":
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError("Optimizer Not Understood: {}".format(op_type))
