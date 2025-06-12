from typing import Protocol

from framework.core.graph import Dataset, Graph


class DatasetFeatureExtractor(Protocol):
    """
    Base class for feature extractors.
    """

    def extract(self, dataset: Dataset) -> Dataset:
        """
        Extract features from the given dataset.

        :param dataset: The dataset from which to extract features.
        :return: A new dataset with extracted features.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GraphFeatureExtractor(Protocol):
    """
    Base class for graph feature extractors.
    """

    def extract_from_graph(self, graph: Graph) -> dict:
        """
        Extract features from the given graph.

        :param graph: The graph from which to extract features.
        :return: A dictionary of extracted features.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class FeatureExtractorCollection:
    """
    A collection of feature extractors.
    """

    def __init__(self, extractors: list[GraphFeatureExtractor]):
        """
        Initializes the collection with a list of feature extractors.

        :param extractors: A list of GraphFeatureExtractor instances.
        """
        self.extractors = extractors

    def extract(self, dataset: Dataset) -> Dataset:
        """
        Extract features from the dataset using all extractors in the collection.

        :param dataset: The dataset from which to extract features.
        :return: A new dataset with extracted features.
        """
        for graph in dataset:
            for extractor in self.extractors:
                features = extractor.extract_from_graph(graph)
                graph.features.update(features)
        return dataset
