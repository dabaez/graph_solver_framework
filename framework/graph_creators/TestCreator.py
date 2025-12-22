import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("test_creator")
class TestGraphCreator:
    def description(self) -> str:
        return "A test graph creator that generates a simple graph."

    def required_parameters(self) -> list[RequiredParameter]:
        return []

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if parameters:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
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
        with dataset.writer() as writer:
            writer.add(create_in_memory_graph(G))
        return dataset
