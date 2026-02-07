import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("ErdosRenyiRandomParametersGenerator")
class ErdosRenyiRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Erdos-Renyi graphs with random parameters kept within specified ranges."

    def required_parameters(self) -> list:
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
                name="min_p",
                description="Minimum probability for edge creation.",
            ),
            RequiredParameter(
                name="max_p",
                description="Maximum probability for edge creation.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, float, float]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        min_p = float(parameters["min_p"])
        max_p = float(parameters["max_p"])
        return number_of_graphs, min_n, max_n, min_p, max_p

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
            or "min_p" not in parameters
            or "max_p" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_n, max_n, min_p, max_p = self.parse_parameters(
                parameters
            )
            if number_of_graphs <= 0:
                return False
            if min_n < 0 or max_n < min_n:
                return False
            if not (0.0 <= min_p <= 1.0) or not (0.0 <= max_p <= 1.0) or max_p < min_p:
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n, max_n, min_p, max_p = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.randint(min_n, max_n)
                p = random.uniform(min_p, max_p)
                graph = nx.erdos_renyi_graph(n, p)
                in_memory_graph = create_in_memory_graph(graph)
                writer.add(in_memory_graph)
        return dataset
