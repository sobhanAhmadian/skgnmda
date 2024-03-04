import abc


class Config(abc.ABC):
    @abc.abstractmethod
    def get_configuration(self):
        pass

    @abc.abstractmethod
    def get_summary(self):
        pass


class ModelConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.embed_dim = None  # dimension of embedding

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
        }

    def get_summary(self):
        return {
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
        }


class OptimizerConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.lr = None  # learning rate
        self.batch_size = None
        self.n_epoch = None
        self.exp_name = None

        # checkpoint configuration
        self.checkpoint_dir = None
        self.checkpoint_monitor = "train_auc"
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = "max"
        self.checkpoint_verbose = 1

        # early_stopping configuration
        self.early_stopping_monitor = "train_auc"
        self.early_stopping_mode = "max"
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3

    def get_configuration(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_monitor": self.checkpoint_monitor,
            "checkpoint_save_best_only": self.checkpoint_save_best_only,
            "checkpoint_save_weights_only": self.checkpoint_save_weights_only,
            "checkpoint_save_weights_mode": self.checkpoint_save_weights_mode,
            "checkpoint_verbose": self.checkpoint_verbose,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_mode": self.early_stopping_mode,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_verbose": self.early_stopping_verbose,
            "callbacks_to_add": self.callbacks_to_add,
            "swa_start": self.swa_start,
        }

    def get_summary(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
            "checkpoint_monitor": self.checkpoint_monitor,
            "early_stopping_monitor": self.early_stopping_monitor,
            "callbacks_to_add": self.callbacks_to_add,
        }
