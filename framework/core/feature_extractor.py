from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import networkx as nx


@dataclass
class Feature:
    name: str
    value: Any


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

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        """
        Extract features from the given graph.

        :param graph: The graph from which to extract features.
        :return: A list of features extracted from the graph.
        """
        ...
