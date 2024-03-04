from .aggregator import SumAggregator, ConcatAggregator, NeighborAggregator, SumConcatAggregator
from .mapping import PairScore

Aggregator = {
    'sum': SumAggregator,
    'concat': ConcatAggregator,
    'neigh': NeighborAggregator,
    'sum_concat': SumConcatAggregator
}
