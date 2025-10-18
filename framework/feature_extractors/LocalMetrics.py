import math
import random
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

        def extract_features(self, graph: FrameworkGraph) -> list[Feature]:
            values = feature_extractor.extract_features(graph)
            n = len(values)
            if n == 0:
                mean = std_dev = min_value = max_value = log_abs_skew = (
                    skew_positive
                ) = log_kurtosis = 0.0
                const = 1.0
            else:
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / n
                std_dev = math.sqrt(variance)

                min_value = min(values)
                max_value = max(values)

                if std_dev == 0:
                    skewness = 0.0
                    kurtosis = 0.0
                else:
                    skewness = (sum((x - mean) ** 3 for x in values) / n) / (std_dev**3)
                    kurtosis = (sum((x - mean) ** 4 for x in values) / n) / (
                        std_dev**4
                    ) - 3

                log_abs_skew = math.log(abs(skewness) + 1e-10)
                skew_positive = 1.0 if skewness > 0 else 0.0
                log_kurtosis = math.log(abs(kurtosis) + 1e-10)
                const = 1.0 if all(x == values[0] for x in values) else 0.0

            return [
                Feature(f"{feature_extractor.name()}_mean", mean),
                Feature(f"{feature_extractor.name()}_std_dev", std_dev),
                Feature(f"{feature_extractor.name()}_min", min_value),
                Feature(f"{feature_extractor.name()}_max", max_value),
                Feature(f"{feature_extractor.name()}_log_abs_skew", log_abs_skew),
                Feature(f"{feature_extractor.name()}_skew_positive", skew_positive),
                Feature(f"{feature_extractor.name()}_log_kurtosis", log_kurtosis),
                Feature(f"{feature_extractor.name()}_const", const),
            ]

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


class ClusteringCoefficient:
    def description(self) -> str:
        return "Extracts the clustering coefficients of the graph."

    def name(self) -> str:
        return "Clustering Coefficient"

    def extract_features(self, graph: FrameworkGraph) -> list[float]:
        n = graph.graph_object.number_of_nodes()
        k = min(n, 3 * math.ceil(math.log(n)))
        all_nodes = list(graph.graph_object.nodes())
        sampled_nodes = random.sample(all_nodes, k)
        return list(nx.clustering(graph.graph_object, sampled_nodes).values())  # type: ignore


ClusteringCoefficientFeatureExtractor = statistical_measures_from_list(
    ClusteringCoefficient()
)


class AverageNeighborDegree:
    def description(self) -> str:
        return "Extracts the average neighbor degrees of the graph."

    def name(self) -> str:
        return "Average Neighbor Degree"

    def extract_features(self, graph: FrameworkGraph) -> list[float]:
        nodes_with_neighbors = [
            n
            for n in graph.graph_object.nodes()
            if len(list(graph.graph_object.neighbors(n))) > 0
        ]
        return nx.average_neighbor_degree(
            graph.graph_object, nodes=nodes_with_neighbors
        ).values()  # type: ignore


AverageNeighborDegreeFeatureExtractor = statistical_measures_from_list(
    AverageNeighborDegree()
)


class CoreNumber:
    def description(self) -> str:
        return "Extracts the core numbers of the graph."

    def name(self) -> str:
        return "Core Number"

    def extract_features(self, graph: FrameworkGraph) -> list[float]:
        return list(nx.core_number(graph.graph_object).values())


CoreNumberFeatureExtractor = statistical_measures_from_list(CoreNumber())
