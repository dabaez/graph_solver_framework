from dataclasses import dataclass
from typing import Any, NewType

from networkx import Graph


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

    def add_feature(self, feature: Feature):
        for i, existing_feature in enumerate(self.features):
            if existing_feature.name == feature.name:
                self.features[i] = feature
                return
        self.features.append(feature)


Dataset = NewType("Dataset", list[FrameworkGraph])
