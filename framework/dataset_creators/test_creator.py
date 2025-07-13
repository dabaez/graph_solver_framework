from typing import Any

import networkx as nx

from framework.core.dataset_creator import RequiredParameter
from framework.core.factories import DatasetFromNetworkX
from framework.core.graph import Dataset
from framework.core.registries import register_dataset_creator


@register_dataset_creator("test_creator")
class TestDatasetCreator:
    def required_parameters(self) -> list[RequiredParameter]:
        return []

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        if parameters:
            return False
        return True

    def create_dataset(self, parameters: dict[str, Any]) -> Dataset:
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
        return DatasetFromNetworkX([G])
