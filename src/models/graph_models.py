import keras.backend as K
import sklearn.metrics as m
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
)

from base.basemodel import BaseModel
from base.basemodel import ModelFactory
from base.data import Data
from base.optimization import Result
from src.config import KGCNModelConfig, DataConfig
from src.models.layers import Aggregator
from src.models.layers.mapping import PairScore


class PairKGCN(BaseModel):
    def __init__(
        self,
        kgcn_config: KGCNModelConfig,
        data_config: DataConfig,
        first_term_size=None,
        second_term_size=None,
    ):
        super(PairKGCN, self).__init__(kgcn_config, data_config)
        self.second_term_size = second_term_size
        self.first_term_size = first_term_size
        self.model = self.build()

    def build(self):
        first_input = Input(shape=(1,), name="first_input", dtype="int64")
        second_input = Input(shape=(1,), name="second_input", dtype="int64")

        if self.first_term_size is not None and self.second_term_size is not None:
            first_term_input = Input(
                shape=(self.first_term_size,), name="first_term_input", dtype="float64"
            )
            second_term_input = Input(
                shape=(self.second_term_size,),
                name="second_term_input",
                dtype="float64",
            )

        first_embedding = Embedding(
            input_dim=self.data_config.entity_vocab_size,
            output_dim=self.model_config.embed_dim,
            embeddings_initializer="glorot_normal",
            embeddings_regularizer=l2(self.model_config.l2_weight),
            name="first_embedding",
        )(first_input)
        second_embedding = Embedding(
            input_dim=self.data_config.entity_vocab_size,
            output_dim=self.model_config.embed_dim,
            embeddings_initializer="glorot_normal",
            embeddings_regularizer=l2(self.model_config.l2_weight),
            name="second_embedding",
        )(second_input)

        entity_embedder = Embedding(
            input_dim=self.data_config.entity_vocab_size,
            output_dim=self.model_config.embed_dim,
            embeddings_initializer="glorot_normal",
            embeddings_regularizer=l2(self.model_config.l2_weight),
            name="entity_embedding",
        )
        relation_embedder = Embedding(
            input_dim=self.data_config.relation_vocab_size,
            output_dim=self.model_config.embed_dim,
            embeddings_initializer="glorot_normal",
            embeddings_regularizer=l2(self.model_config.l2_weight),
            name="relation_embedding",
        )

        neighbor_embedding = Lambda(
            lambda x: self.get_neighbor_info(x[0], x[1], x[2]), name="neigh_embedding"
        )

        adj_entity_matrix = tf.Variable(
            self.data_config.adj_entity,
            name="adj_entity",
            dtype="int64",
            trainable=False,
        )
        adj_relation_matrix = tf.Variable(
            self.data_config.adj_relation,
            name="adj_relation",
            dtype="int64",
            trainable=False,
        )

        first_neigh_ent_list = Lambda(
            lambda x: self.get_receptive_field(
                x, adj_entity_matrix, adj_relation_matrix
            ),
            name="receptive_filed_for_microbe_ent",
        )(first_input)
        first_neigh_rel_list = Lambda(
            lambda x: self.get_receptive_field(
                x, adj_entity_matrix, adj_relation_matrix, True
            ),
            name="receptive_filed_for_microbe_rel",
        )(first_input)

        first_neigh_ent_embed_list = [
            entity_embedder(neigh_ent) for neigh_ent in first_neigh_ent_list
        ]
        first_neigh_rel_embed_list = [
            relation_embedder(neigh_rel) for neigh_rel in first_neigh_rel_list
        ]

        # Iteratively update first_neigh_ent_embed_list
        for depth in range(self.model_config.n_depth):
            aggregator = Aggregator[self.model_config.aggregator_type](
                activation="tanh" if depth == self.model_config.n_depth - 1 else "relu",
                regularizer=l2(self.model_config.l2_weight),
                name=f"first_aggregator_{depth + 1}",
            )
            next_list = []
            for hop in range(self.model_config.n_depth - depth):
                first_neighbor_embed = neighbor_embedding(
                    [
                        first_embedding,
                        first_neigh_rel_embed_list[hop],
                        first_neigh_ent_embed_list[hop + 1],
                    ]
                )
                next_list.append(
                    aggregator([first_neigh_ent_embed_list[hop], first_neighbor_embed])
                )
            first_neigh_ent_embed_list = next_list

        second_neigh_ent_list = Lambda(
            lambda x: self.get_receptive_field(
                x, adj_entity_matrix, adj_relation_matrix
            ),
            name="receptive_filed_for_second_ent",
        )(second_input)
        second_neigh_rel_list = Lambda(
            lambda x: self.get_receptive_field(
                x, adj_entity_matrix, adj_relation_matrix, True
            ),
            name="receptive_filed_for_second_rel",
        )(second_input)

        second_neigh_ent_embed_list = [
            entity_embedder(neigh_ent) for neigh_ent in second_neigh_ent_list
        ]
        second_neigh_rel_embed_list = [
            relation_embedder(neigh_rel) for neigh_rel in second_neigh_rel_list
        ]

        for depth in range(self.model_config.n_depth):
            aggregator = Aggregator[self.model_config.aggregator_type](
                activation="tanh" if depth == self.model_config.n_depth - 1 else "relu",
                regularizer=l2(self.model_config.l2_weight),
                name=f"aggregator_{depth + 1}",
            )
            next_list = []
            for hop in range(self.model_config.n_depth - depth):
                second_neighbor_embed = neighbor_embedding(
                    [
                        second_embedding,
                        second_neigh_rel_embed_list[hop],
                        second_neigh_ent_embed_list[hop + 1],
                    ]
                )
                next_list.append(
                    aggregator(
                        [second_neigh_ent_embed_list[hop], second_neighbor_embed]
                    )
                )
            second_neigh_ent_embed_list = next_list

        final_first_embedding = Lambda(lambda x: K.squeeze(x, axis=1))(
            first_neigh_ent_embed_list[0]
        )
        final_second_embedding = Lambda(lambda x: K.squeeze(x, axis=1))(
            second_neigh_ent_embed_list[0]
        )

        if self.second_term_size is not None and self.first_term_size is not None:
            # first_squeeze_pre_embedding = Lambda(lambda x: K.squeeze(x, axis=1))(first_term_input)
            # second_squeeze_pre_embedding = Lambda(lambda x: K.squeeze(x, axis=1))(second_term_input)

            final_first_embedding = Lambda(lambda x: K.concatenate([x[0], x[1]]))(
                [final_first_embedding, first_term_input]
            )
            final_second_embedding = Lambda(lambda x: K.concatenate([x[0], x[1]]))(
                [final_second_embedding, second_term_input]
            )

        print(final_second_embedding)
        pair_score = PairScore(
            activation="tanh",
            regularizer=l2(self.model_config.l2_weight),
            name="pair_score",
        )(
            [final_second_embedding, final_first_embedding]
        )  # MLP operation

        if self.second_term_size is not None and self.first_term_size is not None:
            model = Model(
                [second_input, first_input, second_term_input, first_term_input],
                pair_score,
            )
        else:
            model = Model([second_input, first_input], pair_score)

        return model

    def destroy(self):
        del self.model
        K.clear_session()

    def predict(self, X):
        return self.model.predict(X).flatten()

    def summary(self):
        self.model.summary()

    def load_weights(self, filename: str):
        self.model.load_weights(filename)

    def evaluate(self, data: Data, thresh=0.5):
        result = Result()

        y_true = data.y.flatten()
        logit = self.predict(data.X)
        y_predicted = tf.nn.sigmoid(logit)
        result.auc = roc_auc_score(y_true=y_true, y_score=y_predicted)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_predicted)
        result.aupr = m.auc(r, p)
        y_predicted = [1 if prob >= thresh else 0 for prob in y_predicted]
        result.acc = accuracy_score(y_true=y_true, y_pred=y_predicted)
        result.f1 = f1_score(y_true=y_true, y_pred=y_predicted)

        return result

    def get_receptive_field(
        self, entity, adj_entity_matrix, adj_relation_matrix, relation=False
    ):
        """Calculate receptive field for entity using adjacent matrix

        param entity: a tensor shaped [batch_size, 1]
        return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neighbor_entity_list = [entity]
        neighbor_relation_list = []
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.model_config.n_depth):
            new_neighbor_entity = K.gather(
                adj_entity_matrix, K.cast(neighbor_entity_list[-1], dtype="int64")
            )  # cast function used to transform data type
            new_neighbor_relation = K.gather(
                adj_relation_matrix, K.cast(neighbor_entity_list[-1], dtype="int64")
            )

            neighbor_entity_list.append(
                K.reshape(new_neighbor_entity, (-1, n_neighbor ** (i + 1)))
            )
            neighbor_relation_list.append(
                K.reshape(new_neighbor_relation, (-1, n_neighbor ** (i + 1)))
            )

        if relation:
            return neighbor_relation_list
        else:
            return neighbor_entity_list

    def get_neighbor_info(self, microbe_embd, relation_embds, entity_embds):
        """Get neighbor representation.

        param microbe_embd: a tensor shaped [batch_size, 1, embed_dim]
        param relation_embds: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        param entity_embds: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] microbe-relation score : C_m_r
        microbe_rel_score = K.relu(
            K.sum(microbe_embd * relation_embds, axis=-1, keepdims=True)
        )

        # [batch_size, neighbor_size ** hop, embed_dim] v_m_neighbour
        weighted_ent = microbe_rel_score * entity_embds

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = K.reshape(
            weighted_ent,
            (
                K.shape(weighted_ent)[0],
                -1,
                self.model_config.neighbor_sample_size,
                self.model_config.embed_dim,
            ),
        )

        # [batch_size, neighbor_size ** (hop-1), embed_dim]
        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed


class PairKGCNFactory(ModelFactory):
    def __init__(
        self,
        kgcn_config: KGCNModelConfig,
        data_config: DataConfig,
        first_term_size,
        second_term_size,
    ):
        self.kgcn_config = kgcn_config
        self.data_config = data_config
        self.first_term_size = first_term_size
        self.second_term_size = second_term_size

    def make_model(self) -> BaseModel:
        model = PairKGCN(
            self.kgcn_config,
            self.data_config,
            self.first_term_size,
            self.second_term_size,
        )
        return model
