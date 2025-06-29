from framework.core.factories import DatasetFeatureExtractorFromGraphFeatureExtractor
from framework.core.graph import Feature, FrameworkGraph


class TestGraphFeatureExtractor:
    def extract_features(self, graph: FrameworkGraph) -> list[Feature]:
        number_of_nodes = Feature(
            name="number_of_nodes", value=graph.graph_object.number_of_nodes()
        )
        number_of_edges = Feature(
            name="number_of_edges", value=graph.graph_object.number_of_edges()
        )
        average_degree = Feature(
            name="average_degree",
            value=graph.graph_object.number_of_edges()
            / graph.graph_object.number_of_nodes()
            if graph.graph_object.number_of_nodes() > 0
            else 0,
        )
        return [number_of_nodes, number_of_edges, average_degree]


DatasetFeatureExtractorFromGraphFeatureExtractor(
    TestGraphFeatureExtractor, "test_graph_feature_extractor"
)
