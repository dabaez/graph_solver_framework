from typing import Protocol, Type

import networkx as nx

from framework.core.feature_extractor import FeatureExtractor
from framework.core.graph import Feature, FrameworkGraph
from framework.core.registries import register_feature_extractor


class LocalMetricsFeatureExtractor(Protocol):
    def description(self) -> str: ...

    def name(self) -> str: ...

    def extract_features(self, graph) -> list[float]: ...


def statistical_measures_from_list(
    feature_extractor: LocalMetricsFeatureExtractor,
) -> Type[FeatureExtractor]:
    @register_feature_extractor(feature_extractor.name())
    class StatisticalMeasures:
        def description(self) -> str:
            return feature_extractor.description()

        def feature_names(self) -> list[str]:
            return [
                f"{feature_extractor.name()}_mean",
                f"{feature_extractor.name()}_std_dev",
                f"{feature_extractor.name()}_min",
                f"{feature_extractor.name()}_max",
                f"{feature_extractor.name()}_log_abs_skew",
                f"{feature_extractor.name()}_skew_positive",
                f"{feature_extractor.name()}_log_kurtosis",
                f"{feature_extractor.name()}_const",
            ]

        def extract_features(self, graph: FrameworkGraph) -> list[Feature]: ...

    return StatisticalMeasures


class AverageDegreeConnectivity:
    def description(self) -> str:
        return "Extracts the average degree connectivity of the graph."

    def name(self) -> str:
        return "Average Degree Connectivity"

    def extract_features(self, graph: FrameworkGraph) -> list[float]:
        return list(nx.average_degree_connectivity(graph.graph_object).values())


AverageDegreeConnectivityFeatureExtractor = statistical_measures_from_list(
    AverageDegreeConnectivity()
)


class NodeDegree:
    def description(self) -> str:
        return "Extracts the node degrees of the graph."

    def name(self) -> str:
        return "Node Degree"

    def extract_features(self, graph: FrameworkGraph) -> list[float]:
        return [d for n, d in graph.graph_object.degree()]  # type: ignore


NodeDegreeFeatureExtractor = statistical_measures_from_list(NodeDegree())
