from typing import Any, Iterator

import networkx as nx

from framework.core.graph import BatchWriter, Dataset, FrameworkGraph, Update


class InMemoryGraphLoader:
    _graph: nx.Graph

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def load(self) -> nx.Graph:
        return self._graph

    def unload(self) -> None:
        pass


def create_in_memory_graph(
    graph: nx.Graph, features: dict[str, Any] | None = None
) -> FrameworkGraph:
    if features is None:
        features = {}
    loader = InMemoryGraphLoader(graph)
    return FrameworkGraph(id=-1, features=features, loader=loader)


class MemoryDataset:
    _graphs: list[FrameworkGraph]

    def __init__(self):
        self._graphs = []

    def __getitem__(self, index: int) -> FrameworkGraph:
        return self._graphs[index]

    def __len__(self) -> int:
        return len(self._graphs)

    def __iter__(self) -> Iterator[FrameworkGraph]:
        return iter(self._graphs)

    def append(self, graph: FrameworkGraph) -> None:
        features = graph.features.copy()
        with graph as g:
            new_graph = create_in_memory_graph(g, features)
        self._graphs.append(new_graph)

    def extend(self, dataset: Dataset) -> None:
        for graph in dataset:
            self.append(graph)

    def writer(self, batch_size: int = 1000) -> BatchWriter:
        return BatchWriter(
            save_callback=self._batch_save_callback, batch_size=batch_size
        )

    def _batch_save_callback(self, updates: list[Update]) -> None:
        for update in updates:
            if update.update_type == "add":
                self.append(update.graph)
