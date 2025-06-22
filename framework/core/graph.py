from dataclasses import dataclass
from typing import Any, NewType

from networkx import Graph

MaximumIndependentSet = NewType("MaximumIndependentSet", list[str])


@dataclass
class Feature:
    name: str
    value: Any


@dataclass
class FrameworkGraph:
    graph_object: Graph
    features: list[Feature] = []


Dataset = NewType("Dataset", list[FrameworkGraph])


def DatasetFromNetworkX(graphs: list[Graph]) -> Dataset:
    return Dataset([FrameworkGraph(graph_object=graph) for graph in graphs])
