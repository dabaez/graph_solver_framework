import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("BipartiteRandomParametersGenerator")
class BipartiteRandomParametersGenerator:
    def description(self) -> str:
        return (
            "Generates Bipartite Graphs with random parameters within specified ranges."
        )

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="number_of_graphs",
                description="Number of graphs to generate.",
            ),
            RequiredParameter(
                name="min_n1",
                description="Minimum number of nodes in the first partition.",
            ),
            RequiredParameter(
                name="max_n1",
                description="Maximum number of nodes in the first partition.",
            ),
            RequiredParameter(
                name="min_n2",
                description="Minimum number of nodes in the second partition.",
            ),
            RequiredParameter(
                name="max_n2",
                description="Maximum number of nodes in the second partition.",
            ),
            RequiredParameter(
                name="min_p",
                description="Minimum probability of edge creation between partitions.",
            ),
            RequiredParameter(
                name="max_p",
                description="Maximum probability of edge creation between partitions.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int, float, float]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n1 = int(parameters["min_n1"])
        max_n1 = int(parameters["max_n1"])
        min_n2 = int(parameters["min_n2"])
        max_n2 = int(parameters["max_n2"])
        min_p = float(parameters["min_p"])
        max_p = float(parameters["max_p"])
        return number_of_graphs, min_n1, max_n1, min_n2, max_n2, min_p, max_p

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n1" not in parameters
            or "max_n1" not in parameters
            or "min_n2" not in parameters
            or "max_n2" not in parameters
            or "min_p" not in parameters
            or "max_p" not in parameters
        ):
            return False
        try:
            (
                number_of_graphs,
                min_n1,
                max_n1,
                min_n2,
                max_n2,
                min_p,
                max_p,
            ) = self.parse_parameters(parameters)
        except ValueError:
            return False
        if (
            number_of_graphs <= 0
            or min_n1 <= 0
            or max_n1 < min_n1
            or min_n2 <= 0
            or max_n2 < min_n2
            or min_p < 0.0
            or max_p > 1.0
            or max_p < min_p
        ):
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n1, max_n1, min_n2, max_n2, min_p, max_p = (
            self.parse_parameters(parameters)
        )
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n1 = random.randint(min_n1, max_n1)
                n2 = random.randint(min_n2, max_n2)
                p = random.uniform(min_p, max_p)
                G = nx.bipartite.random_graph(n1, n2, p)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
