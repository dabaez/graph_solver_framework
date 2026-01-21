import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("StarRandomParametersGenerator")
class StarRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Star Graphs with random parameters within specified ranges."

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="number_of_graphs",
                description="Number of graphs to generate.",
            ),
            RequiredParameter(
                name="min_n",
                description="Minimum number of leaves in the star graph.",
            ),
            RequiredParameter(
                name="max_n",
                description="Maximum number of leaves in the star graph.",
            ),
        ]

    def parse_parameters(self, parameters: dict[str, str]) -> tuple[int, int, int]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        return number_of_graphs, min_n, max_n

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_n, max_n = self.parse_parameters(parameters)
            if (
                number_of_graphs <= 0
                or min_n <= 0
                or max_n < min_n
                or number_of_graphs > (max_n - min_n + 1)
            ):
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n, max_n = self.parse_parameters(parameters)
        non_added_sizes = [_ for _ in range(min_n, max_n + 1)]
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.choice(non_added_sizes)
                non_added_sizes.remove(n)
                G = nx.star_graph(n)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
