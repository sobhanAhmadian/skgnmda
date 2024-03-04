from keras import backend as K
from keras.layers import Layer


class AggregatorLayer(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(AggregatorLayer, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SumAggregator(AggregatorLayer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(SumAggregator, self).__init__(activation, initializer, regularizer, **kwargs)
        self.w = None
        self.b = None

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        self.w = self.add_weight(name=self.name + '_w', shape=(ent_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)  # 为该层定义一个可训练的权重
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,), initializer='zeros')
        super(SumAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot((entity + neighbor), self.w) + self.b)


class ConcatAggregator(AggregatorLayer):

    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(ConcatAggregator, self).__init__(activation, initializer, regularizer, **kwargs)
        self.w = None
        self.b = None

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        neighbor_embed_dim = input_shape[1][-1]
        self.w = self.add_weight(name=self.name + '_w',
                                 shape=(ent_embed_dim + neighbor_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
                                 initializer='zeros')
        super(ConcatAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot(K.concatenate([entity, neighbor]), self.w) + self.b)


class SumConcatAggregator(AggregatorLayer):

    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(SumConcatAggregator, self).__init__(activation, initializer, regularizer, **kwargs)
        self.w_sum = None
        self.b_sum = None
        self.w_concat = None
        self.b_concat = None

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        self.w_sum = self.add_weight(name=self.name + '_w_sum',
                                     shape=(ent_embed_dim, ent_embed_dim),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer)
        self.b_sum = self.add_weight(name=self.name + '_b_sum',
                                     shape=(ent_embed_dim,),
                                     initializer='zeros')
        neighbor_embed_dim = input_shape[1][-1]
        self.w_concat = self.add_weight(name=self.name + '_w_concat',
                                        shape=(ent_embed_dim + neighbor_embed_dim, ent_embed_dim),
                                        initializer=self.initializer,
                                        regularizer=self.regularizer)
        self.b_concat = self.add_weight(name=self.name + '_b_concat',
                                        shape=(ent_embed_dim,),
                                        initializer='zeros')
        super(SumConcatAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot((entity + neighbor), self.w_sum) + self.b_sum) + self.activation(
            K.dot(K.concatenate([entity, neighbor]), self.w_concat) + self.b_concat)


class NeighborAggregator(AggregatorLayer):

    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(NeighborAggregator, self).__init__(activation, initializer, regularizer, **kwargs)
        self.w = None
        self.b = None

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        neighbor_embed_dim = input_shape[1][-1]
        self.w = self.add_weight(name=self.name + '_w',
                                 shape=(neighbor_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
                                 initializer='zeros')
        super(NeighborAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot(neighbor, self.w) + self.b)
