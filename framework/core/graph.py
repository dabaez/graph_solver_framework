from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Protocol, TypeVar

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
    ) -> bool:
        """Adds a feature to the graph.

        :param feature_name: Name of the feature.
        :param feature_value: Value of the feature.
        :param overwrite: If True, overwrites the feature if it already exists.

        :return: True if the feature was added or updated, False otherwise.
        """
        if (feature_name not in self.features) or (
            self.features[feature_name] != feature_value and overwrite
        ):
            self.features[feature_name] = feature_value
            return True
        return False


@dataclass
class Update:
    update_type: Literal["add", "feature_update"]
    graph: FrameworkGraph


class BatchWriter:
    def __init__(
        self, save_callback: Callable[[list[Update]], None], batch_size: int = 1000
    ):
        self.save_callback = save_callback
        self.batch_size = batch_size
        self.updates: list[Update] = []

    def add(self, graph: FrameworkGraph) -> None:
        self.updates.append(Update(update_type="add", graph=graph))
        self._check_flush()

    def update_features(self, graph: FrameworkGraph) -> None:
        self.updates.append(Update(update_type="feature_update", graph=graph))
        self._check_flush()

    def _check_flush(self) -> None:
        if len(self.updates) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if self.updates:
            self.save_callback(self.updates)
            self.updates = []

    def __enter__(self) -> "BatchWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._flush()


D = TypeVar("D", bound="Dataset")


class Dataset(Protocol):
    def __getitem__(self, index: int) -> FrameworkGraph:
        """Returns the FrameworkGraph at the specified index."""
        ...

    def __len__(self) -> int:
        """Returns the number of FrameworkGraphs in the dataset."""
        ...

    def __iter__(self) -> Iterator[FrameworkGraph]:
        """Returns an iterator over the FrameworkGraphs in the dataset."""
        ...

    def writer(self, batch_size: int = 1000) -> BatchWriter:
        """Returns a Writer for updating graphs in the dataset."""
        ...
