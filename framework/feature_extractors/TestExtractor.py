import networkx as nx

from framework.core.feature_extractor import Feature
from framework.core.registries import register_feature_extractor


@register_feature_extractor("test_graph_feature_extractor")
class TestGraphFeatureExtractor:
    def description(self) -> str:
        return "Extracts basic graph features. Includes number of nodes, number of edges, and average degree."

    def feature_names(self) -> list[str]:
        return ["number_of_nodes", "number_of_edges", "average_degree"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        number_of_nodes = Feature(name="number_of_nodes", value=graph.number_of_nodes())
        number_of_edges = Feature(name="number_of_edges", value=graph.number_of_edges())
        average_degree = Feature(
            name="average_degree",
            value=graph.number_of_edges() / graph.number_of_nodes()
            if graph.number_of_nodes() > 0
            else 0,
        )
        return [number_of_nodes, number_of_edges, average_degree]
