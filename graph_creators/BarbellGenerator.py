import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("BarbellRandomParametersGenerator")
class BarbellRandomParametersGenerator:
    def description(self) -> str:
        return (
            "Generates Barbell Graphs with random parameters within specified ranges."
        )

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="number_of_graphs",
                description="Number of graphs to generate.",
            ),
            RequiredParameter(
                name="min_m1",
                description="Minimum number of nodes in each complete graph.",
            ),
            RequiredParameter(
                name="max_m1",
                description="Maximum number of nodes in each complete graph.",
            ),
            RequiredParameter(
                name="min_m2",
                description="Minimum number of nodes in the path connecting the two complete graphs.",
            ),
            RequiredParameter(
                name="max_m2",
                description="Maximum number of nodes in the path connecting the two complete graphs.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_m1 = int(parameters["min_m1"])
        max_m1 = int(parameters["max_m1"])
        min_m2 = int(parameters["min_m2"])
        max_m2 = int(parameters["max_m2"])
        return number_of_graphs, min_m1, max_m1, min_m2, max_m2

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_m1" not in parameters
            or "max_m1" not in parameters
            or "min_m2" not in parameters
            or "max_m2" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_m1, max_m1, min_m2, max_m2 = self.parse_parameters(
                parameters
            )
            if (
                number_of_graphs <= 0
                or min_m1 <= 0
                or max_m1 < min_m1
                or min_m2 < 0
                or max_m2 < min_m2
            ):
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_m1, max_m1, min_m2, max_m2 = self.parse_parameters(
            parameters
        )
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                m1 = random.randint(min_m1, max_m1)
                m2 = random.randint(min_m2, max_m2)
                G = nx.barbell_graph(m1, m2)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
