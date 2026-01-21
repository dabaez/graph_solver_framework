import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("RandomGeometricRandomParametersGenerator")
class RandomGeometricRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Random Geometric Graphs with random parameters within specified ranges."

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="number_of_graphs",
                description="Number of graphs to generate.",
            ),
            RequiredParameter(
                name="min_n",
                description="Minimum number of nodes in the graph.",
            ),
            RequiredParameter(
                name="max_n",
                description="Maximum number of nodes in the graph.",
            ),
            RequiredParameter(
                name="min_radius",
                description="Minimum radius for connecting nodes.",
            ),
            RequiredParameter(
                name="max_radius",
                description="Maximum radius for connecting nodes.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, float, float]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        min_radius = float(parameters["min_radius"])
        max_radius = float(parameters["max_radius"])
        return number_of_graphs, min_n, max_n, min_radius, max_radius

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
            or "min_radius" not in parameters
            or "max_radius" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_n, max_n, min_radius, max_radius = (
                self.parse_parameters(parameters)
            )
        except ValueError:
            return False
        if number_of_graphs <= 0:
            return False
        if min_n <= 0 or max_n < min_n:
            return False
        if min_radius < 0 or max_radius < min_radius:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n, max_n, min_radius, max_radius = self.parse_parameters(
            parameters
        )
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.randint(min_n, max_n)
                radius = random.uniform(min_radius, max_radius)
                G = nx.random_geometric_graph(n, radius)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
