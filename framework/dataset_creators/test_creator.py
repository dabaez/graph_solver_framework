import networkx as nx

from framework.core.dataset_creator import RequiredParameter
from framework.core.graph import Dataset, FrameworkGraph
from framework.core.registries import register_dataset_creator


@register_dataset_creator("test_creator")
class TestDatasetCreator:
    def required_parameters(self) -> list[RequiredParameter]:
        return []

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if parameters:
            return False
        return True

    def create_dataset(self, parameters: dict[str, str]) -> Dataset:
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C", "D", "E"])
        G.add_edges_from(
            [
                ("A", "B"),
                ("A", "E"),
                ("E", "B"),
                ("E", "D"),
                ("D", "B"),
                ("B", "C"),
                ("C", "D"),
            ]
        )
        return Dataset([FrameworkGraph(G)])
