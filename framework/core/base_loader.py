import os
from typing import Protocol

from framework.core.graph import Dataset, Graph


class DatasetLoader(Protocol):
    """
    Base class for loaders.
    """

    def load(self, path: str | None) -> Dataset:
        """
        Load the data from the given path.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GraphLoader(Protocol):
    """
    Base class for graph loaders.
    """

    def load(self, path: str | None) -> Graph:
        """
        Load the graph from the given path.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class DatasetLoaderFromGraphLoader:
    """
    A loader that uses a GraphLoader to load graphs and create a Dataset.
    """

    def __init__(self, graph_loader: GraphLoader):
        """
        Initializes the loader with a GraphLoader instance.

        :param graph_loader: An instance of GraphLoader.
        """
        self.graph_loader = graph_loader

    def load(self, path: str | None) -> Dataset:
        """
        Load the dataset by loading graphs from the given path.

        :param path: The path from which to load the dataset.
        :return: A Dataset containing the loaded graphs.
        """
        if path is None:
            raise ValueError("Path cannot be None")
        dataset = Dataset(name=os.path.basename(path), graphs=[])
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                graph = self.graph_loader.load(file_path)
                dataset.add_graph(graph)
        return dataset
