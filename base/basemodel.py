import abc

from base.data import Data


class BaseModel(abc.ABC):
    def __init__(self, model_config, data_config) -> None:
        self.model_config = model_config
        self.data_config = data_config
        self.model = None

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError

    @abc.abstractmethod
    def destroy(self):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abc.abstractmethod
    def summary(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, data: Data, thresh):
        raise NotImplementedError


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def make_model(self) -> BaseModel:
        raise NotImplementedError
