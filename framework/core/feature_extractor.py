from typing import Protocol

from framework.core.graph import Dataset, Feature, Graph


class DatasetFeatureExtractor(Protocol):
    """Base class for dataset feature extractors."""

    def extract_features(self, dataset: Dataset) -> list[list[Feature]]:
        """
        Extract features from the dataset.

        :param dataset: The dataset from which to extract features.
        :return: A list of lists of features extracted from the dataset.
        """
        ...


class GraphFeatureExtractor(Protocol):
    """Base class for graph feature extractors."""

    def extract_features(self, graph: Graph) -> list[Feature]:
        """
        Extract features from the graph.

        :param graph: The graph from which to extract features.
        :return: A list of features extracted from the graph.
        """
        ...


class DatasetFeatureExtractorFromGraphFeatureExtractor:
    """
    A feature extractor that uses a GraphFeatureExtractor to extract features from graphs in a dataset.
    """

    def __init__(self, graph_feature_extractor: GraphFeatureExtractor):
        """
        Initializes the extractor with a GraphFeatureExtractor instance.

        :param graph_feature_extractor: An instance of GraphFeatureExtractor.
        """
        self.graph_feature_extractor = graph_feature_extractor

    def extract_features(self, dataset: Dataset) -> list[list[Feature]]:
        """
        Extract features from each graph in the dataset.

        :param dataset: The dataset from which to extract features.
        :return: A list of lists of features extracted from each graph in the dataset.
        """
        return [
            self.graph_feature_extractor.extract_features(graph) for graph in dataset
        ]
