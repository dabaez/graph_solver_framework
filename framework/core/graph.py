from dataclasses import dataclass
from typing import Any, NewType

from networkx import Graph


@dataclass
class Feature:
    name: str
    value: Any


class FrameworkGraph:
    graph_object: Graph
    features: dict[str, Any] = {}

    def __init__(self, graph_object: Graph):
        self.graph_object = graph_object

    def add_feature(self, feature: Feature, overwrite: bool) -> None:
        if overwrite or (feature.name not in self.features):
            self.features[feature.name] = feature.value


Dataset = NewType("Dataset", list[FrameworkGraph])
