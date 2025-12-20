import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_dataset_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_dataset_creator("test_creator")
class TestDatasetCreator:
    def description(self) -> str:
        return "A test dataset creator that generates a simple graph."

    def required_parameters(self) -> list[RequiredParameter]:
        return []

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if parameters:
            return False
        return True

    def create_dataset(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
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
        dataset.append(create_in_memory_graph(G))
        return dataset
