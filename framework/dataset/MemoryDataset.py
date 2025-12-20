from typing import Any

import networkx as nx

from framework.core.graph import FrameworkGraph, Writer


class InMemoryGraphLoader:
    _graph: nx.Graph

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def load(self) -> nx.Graph:
        return self._graph

    def unload(self) -> None:
        pass


def create_in_memory_graph(
    cls, graph: nx.Graph, features: dict[str, Any] | None = None
) -> FrameworkGraph:
    if features is None:
        features = {}
    loader = InMemoryGraphLoader(graph)
    return cls(id=-1, features=features, loader=loader)


class MemoryWriter:
    def update(self, graph: FrameworkGraph) -> None:
        pass

    def __enter__(self) -> "MemoryWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class MemoryDataset:
    _graphs: list[FrameworkGraph]

    def __init__(self):
        self._graphs = []

    def __getitem__(self, index: int) -> FrameworkGraph:
        return self._graphs[index]

    def __len__(self) -> int:
        return len(self._graphs)

    def __iter__(self):
        return iter(self._graphs)

    def append(self, graph: FrameworkGraph) -> None:
        self._graphs.append(graph)

    def extend(self, graphs: list[FrameworkGraph]) -> None:
        self._graphs.extend(graphs)

    def writer(self) -> Writer:
        return MemoryWriter()
