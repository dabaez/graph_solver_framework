import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("LollipopRandomParametersGenerator")
class LollipopRandomParametersGenerator:
    def description(self) -> str:
        return (
            "Generates Lollipop Graphs with random parameters within specified ranges."
        )

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="number_of_graphs",
                description="Number of graphs to generate.",
            ),
            RequiredParameter(
                name="min_m",
                description="Minimum number of nodes in the complete graph.",
            ),
            RequiredParameter(
                name="max_m",
                description="Maximum number of nodes in the complete graph.",
            ),
            RequiredParameter(
                name="min_n",
                description="Minimum number of nodes in the path.",
            ),
            RequiredParameter(
                name="max_n",
                description="Maximum number of nodes in the path.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_m = int(parameters["min_m"])
        max_m = int(parameters["max_m"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        return number_of_graphs, min_m, max_m, min_n, max_n

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_m" not in parameters
            or "max_m" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_m, max_m, min_n, max_n = self.parse_parameters(
                parameters
            )
            if (
                number_of_graphs <= 0
                or min_m <= 0
                or max_m < min_m
                or min_n < 0
                or max_n < min_n
            ):
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_m, max_m, min_n, max_n = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                m = random.randint(min_m, max_m)
                n = random.randint(min_n, max_n)
                G = nx.lollipop_graph(m, n)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
