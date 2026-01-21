import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("WattsStrogatzGenerator")
class WattsStrogatzGenerator:
    def description(self) -> str:
        return "Generates Watts-Strogatz graphs with specified parameters."

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="min n",
                description="Minimum number of nodes in the graph.",
            ),
            RequiredParameter(
                name="max n",
                description="Maximum number of nodes in the graph.",
            ),
            RequiredParameter(
                name="step n",
                description="Step size for the number of nodes in the graph. If not provided, it defaults to 1.",
            ),
            RequiredParameter(
                name="k",
                description="Each node is joined with its k nearest neighbors in a ring topology. Must be even and less than n.",
            ),
            RequiredParameter(
                name="p",
                description="The probability of rewiring each edge.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, float]:
        min_n = int(parameters["min n"])
        max_n = int(parameters["max n"])
        step_n = int(parameters["step n"]) if parameters.get("step n") else 1
        k = int(parameters["k"])
        p = float(parameters["p"])
        return min_n, max_n, step_n, k, p

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "min n" not in parameters
            or "max n" not in parameters
            or "k" not in parameters
            or "p" not in parameters
        ):
            return False
        try:
            min_n, max_n, step_n, k, p = self.parse_parameters(parameters)
        except ValueError:
            return False
        valid = 0
        for n in range(min_n, max_n + 1, step_n):
            if k < n and k % 2 == 0 and 0 <= p <= 1:
                valid += 1
        return valid > 0

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        min_n, max_n, step_n, k, p = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for n in range(min_n, max_n + 1, step_n):
                G = nx.watts_strogatz_graph(n, k, p)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset


@register_graph_creator("WattsStrogatzRandomParametersGenerator")
class WattsStrogatzRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Watts-Strogatz graphs with random parameters within specified ranges."

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
                name="min_k",
                description="Minimum number of nearest neighbors each node is joined with. Must be even and less than n.",
            ),
            RequiredParameter(
                name="max_k",
                description="Maximum number of nearest neighbors each node is joined with. Must be even and less than n.",
            ),
            RequiredParameter(
                name="min_p",
                description="Minimum probability of rewiring each edge.",
            ),
            RequiredParameter(
                name="max_p",
                description="Maximum probability of rewiring each edge.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int, float, float]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        min_k = int(parameters["min_k"])
        max_k = int(parameters["max_k"])
        min_p = float(parameters["min_p"])
        max_p = float(parameters["max_p"])
        return number_of_graphs, min_n, max_n, min_k, max_k, min_p, max_p

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
            or "min_k" not in parameters
            or "max_k" not in parameters
            or "min_p" not in parameters
            or "max_p" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_n, max_n, min_k, max_k, min_p, max_p = (
                self.parse_parameters(parameters)
            )
            if number_of_graphs <= 0:
                return False
            if min_n < 0 or max_n < min_n:
                return False
            if min_k < 1 or max_k < min_k or min_k > max_n:
                return False
            if min_p < 0 or max_p < min_p or max_p > 1:
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n, max_n, min_k, max_k, min_p, max_p = (
            self.parse_parameters(parameters)
        )
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.randint(min_n, max_n)
                if min_k >= n:
                    k = n - 1
                else:
                    k = random.randint(min_k, min(max_k, n - 1))
                p = random.uniform(min_p, max_p)
                G = nx.watts_strogatz_graph(n, k, p)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
