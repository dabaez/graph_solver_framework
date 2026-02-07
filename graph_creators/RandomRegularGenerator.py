import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("RegularRandomParametersGenerator")
class RegularRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Random Regular Graphs with random parameters within specified ranges."

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="number_of_graphs",
                description="Number of graphs to generate.",
            ),
            RequiredParameter(
                name="min_n",
                description="Minimum number of nodes in the regular graph.",
            ),
            RequiredParameter(
                name="max_n",
                description="Maximum number of nodes in the regular graph.",
            ),
            RequiredParameter(
                name="min_d",
                description="Minimum degree of each node in the regular graph.",
            ),
            RequiredParameter(
                name="max_d",
                description="Maximum degree of each node in the regular graph.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        min_d = int(parameters["min_d"])
        max_d = int(parameters["max_d"])
        return number_of_graphs, min_n, max_n, min_d, max_d

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
            or "min_d" not in parameters
            or "max_d" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_n, max_n, min_d, max_d = self.parse_parameters(
                parameters
            )
            if (
                number_of_graphs <= 0
                or min_n <= 0
                or max_n < min_n
                or min_d < 0
                or max_d < min_d
                or min_d >= max_n
            ):
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n, max_n, min_d, max_d = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.randint(min_n, max_n)
                d = random.randint(min_d, max_d)
                while n * d % 2 != 0 or d >= n:
                    n = random.randint(min_n, max_n)
                    d = random.randint(min_d, max_d)
                G = nx.random_regular_graph(d, n)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
