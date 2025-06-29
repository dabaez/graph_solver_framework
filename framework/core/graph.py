from dataclasses import dataclass
from typing import Any, NewType

from networkx import Graph

MaximumIndependentSet = NewType("MaximumIndependentSet", list[str])


@dataclass
class Feature:
    name: str
    value: Any


class FrameworkGraph:
    graph_object: Graph
    features: list[Feature] = []

    def __init__(self, graph_object: Graph, features: list[Feature] | None = None):
        self.graph_object = graph_object
        if features is not None:
            self.features = features


Dataset = NewType("Dataset", list[FrameworkGraph])
