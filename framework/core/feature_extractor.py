from typing import Protocol, runtime_checkable

from framework.core.graph import Feature, FrameworkGraph


@runtime_checkable
class FeatureExtractor(Protocol):
    """Base class for feature extractors."""

    def description(self) -> str:
        """
        Description of the feature extractor.

        :return: A string describing the feature extractor.
        """
        ...

    def feature_names(self) -> list[str]:
        """
        Get the names of the features extracted by this extractor.

        :return: A list of feature names.
        """
        ...

    def extract_features(self, graph: FrameworkGraph) -> list[Feature]:
        """
        Extract features from the given graph.

        :param graph: The graph from which to extract features.
        :return: A list of features extracted from the graph.
        """
        ...
