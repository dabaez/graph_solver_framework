from typing import Any, Protocol

import networkx as nx


class GraphLoader(Protocol):
    def load(self) -> nx.Graph:
        """Returns the NetworkX graph loaded from the source."""
        ...

    def unload(self) -> None:
        """Cleans up any resources used by the loader."""
        ...


class FrameworkGraph:
    id: int
    features: dict[str, Any]
    _loader: GraphLoader

    def __init__(self, id: int, features: dict[str, Any], loader: GraphLoader):
        self.id = id
        self.features = features
        self._loader = loader

    def __enter__(self) -> nx.Graph:
        return self._loader.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._loader.unload()

    def add_feature(
        self, feature_name: str, feature_value: Any, overwrite: bool = True
    ) -> None:
        """Adds a feature to the graph.

        :param feature_name: Name of the feature.
        :param feature_value: Value of the feature.
        :param overwrite: If True, overwrites the feature if it already exists.
        """
        if overwrite or feature_name not in self.features:
            self.features[feature_name] = feature_value


class Writer(Protocol):
    def update(self, graph: FrameworkGraph) -> None:
        """Updates the given FrameworkGraph in the storage."""
        ...

    def __enter__(self) -> "Writer":
        """Enters the context for writing."""
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exits the context for writing."""
        ...


class Dataset(Protocol):
    def __getitem__(self, index: int) -> FrameworkGraph:
        """Returns the FrameworkGraph at the specified index."""
        ...

    def __len__(self) -> int:
        """Returns the number of FrameworkGraphs in the dataset."""
        ...

    def __iter__(self):
        """Returns an iterator over the FrameworkGraphs in the dataset."""
        ...

    def append(self, graph: FrameworkGraph) -> None:
        """Appends a FrameworkGraph to the dataset."""
        ...

    def extend(self, graphs: list[FrameworkGraph]) -> None:
        """Extends the dataset with a list of FrameworkGraphs."""
        ...

    def writer(self) -> Writer:
        """Returns a Writer for updating graphs in the dataset."""
        ...
