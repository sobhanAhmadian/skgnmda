from keras import backend as K
from keras.layers import Layer


class MappingLayer(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(MappingLayer, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer
        self.embed_dim = 0

    def compute_output_shape(self, input_shape):
        first_shape, second_shape = input_shape
        return first_shape[0], self.embed_dim


class PairScore(MappingLayer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 embed_dim=32, **kwargs):
        super(PairScore, self).__init__(activation, initializer, regularizer, **kwargs)
        self.embed_dim = embed_dim
        self.w_f = None
        self.w_s = None
        self.b_fs = None

    def build(self, input_shape):
        first_embed_dim = input_shape[0][-1]
        second_embed_dim = input_shape[1][-1]

        self.w_f = self.add_weight(name=self.name + '_w_f',
                                   shape=(first_embed_dim, self.embed_dim),
                                   initializer=self.initializer, regularizer=self.regularizer)
        self.w_s = self.add_weight(name=self.name + '_w_s',
                                   shape=(second_embed_dim, self.embed_dim),
                                   initializer=self.initializer, regularizer=self.regularizer)
        self.b_fs = self.add_weight(name=self.name + '_b_fs', shape=(self.embed_dim,), initializer='zeros')
        super(PairScore, self).build(input_shape)

    def call(self, inputs, **kwargs):
        first, second = inputs
        score = K.dot(first, self.w_f) * K.dot(second, self.w_s) + self.b_fs
        score = K.sum(score, axis=1, keepdims=True)
        return score
