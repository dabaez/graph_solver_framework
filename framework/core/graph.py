from dataclasses import dataclass
from typing import Any, NewType

from networkx import Graph as NetworkXGraph

MaximumIndependentSet = NewType("MaximumIndependentSet", list[str])


@dataclass
class Feature:
    name: str
    value: Any


@dataclass
class Graph:
    graph_object: NetworkXGraph
    features: list[Feature] = []


Dataset = NewType("Dataset", list[Graph])
