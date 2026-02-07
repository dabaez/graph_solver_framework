import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("BarabasiAlbertGenerator")
class BarabasiAlbertGenerator:
    def description(self) -> str:
        return "Generates Barabasi-Albert graphs with specified parameters."

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
                name="min m",
                description="Minimum number of edges to attach from a new node to existing nodes.",
            ),
            RequiredParameter(
                name="max m",
                description="Maximum number of edges to attach from a new node to existing nodes. If there is a pair[n, m] where 1 <= m < n isn't satisfied, it will be ignored.",
            ),
            RequiredParameter(
                name="step m",
                description="Step size for the number of edges to attach from a new node to existing nodes. If not provided, it defaults to 1.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int, int]:
        min_n = int(parameters["min n"])
        max_n = int(parameters["max n"])
        min_m = int(parameters["min m"])
        max_m = int(parameters["max m"])
        step_n = int(parameters["step n"]) if parameters.get("step n") else 1
        step_m = int(parameters["step m"]) if parameters.get("step m") else 1
        return min_n, max_n, min_m, max_m, step_n, step_m

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "min n" not in parameters
            or "max n" not in parameters
            or "min m" not in parameters
            or "max m" not in parameters
        ):
            return False
        try:
            min_n, max_n, min_m, max_m, step_n, step_m = self.parse_parameters(
                parameters
            )
        except ValueError:
            return False
        valid = 0
        for n in range(min_n, max_n + 1, step_n):
            for m in range(min_m, max_m + 1, step_m):
                if 1 <= m < n:
                    valid += 1
        return valid > 0

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        min_n, max_n, min_m, max_m, step_n, step_m = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for n in range(min_n, max_n + 1, step_n):
                for m in range(min_m, max_m + 1, step_m):
                    if 1 <= m < n:
                        G = nx.barabasi_albert_graph(n, m)
                        graph = create_in_memory_graph(G)
                        writer.add(graph)
        return dataset


@register_graph_creator("BarabasiAlbertRandomParametersGenerator")
class BarabasiAlbertRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Barabasi-Albert graphs with random parameters kept within specified ranges."

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
                name="min_m",
                description="Minimum number of edges to attach from a new node to existing nodes.",
            ),
            RequiredParameter(
                name="max_m",
                description="Maximum number of edges to attach from a new node to existing nodes.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        min_m = int(parameters["min_m"])
        max_m = int(parameters["max_m"])
        return number_of_graphs, min_n, max_n, min_m, max_m

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
            or "min_m" not in parameters
            or "max_m" not in parameters
        ):
            return False
        try:
            number_of_graphs, min_n, max_n, min_m, max_m = self.parse_parameters(
                parameters
            )
            if number_of_graphs <= 0:
                return False
            if min_n < 0 or max_n < min_n:
                return False
            if min_m < 1 or max_m < min_m:
                return False
        except ValueError:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        number_of_graphs, min_n, max_n, min_m, max_m = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.randint(min_n, max_n)
                m = random.randint(min_m, max_m)
                if 1 <= m < n:
                    G = nx.barabasi_albert_graph(n, m)
                    graph = create_in_memory_graph(G)
                    writer.add(graph)
        return dataset
