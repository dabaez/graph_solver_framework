import random

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("SBMRandomParametersGenerator")
class SBMRandomParametersGenerator:
    def description(self) -> str:
        return "Generates Stochastic Block Model graphs with random parameters within specified ranges."

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
                description="Minimum number of communities in the graph.",
            ),
            RequiredParameter(
                name="max_k",
                description="Maximum number of communities in the graph.",
            ),
            RequiredParameter(
                name="min_p_in",
                description="Minimum probability of edges within communities.",
            ),
            RequiredParameter(
                name="max_p_in",
                description="Maximum probability of edges within communities.",
            ),
            RequiredParameter(
                name="min_p_out",
                description="Minimum probability of edges between communities.",
            ),
            RequiredParameter(
                name="max_p_out",
                description="Maximum probability of edges between communities.",
            ),
        ]

    def parse_parameters(
        self, parameters: dict[str, str]
    ) -> tuple[int, int, int, int, int, float, float, float, float]:
        number_of_graphs = int(parameters["number_of_graphs"])
        min_n = int(parameters["min_n"])
        max_n = int(parameters["max_n"])
        min_k = int(parameters["min_k"])
        max_k = int(parameters["max_k"])
        min_p_in = float(parameters["min_p_in"])
        max_p_in = float(parameters["max_p_in"])
        min_p_out = float(parameters["min_p_out"])
        max_p_out = float(parameters["max_p_out"])
        return (
            number_of_graphs,
            min_n,
            max_n,
            min_k,
            max_k,
            min_p_in,
            max_p_in,
            min_p_out,
            max_p_out,
        )

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if (
            "number_of_graphs" not in parameters
            or "min_n" not in parameters
            or "max_n" not in parameters
            or "min_k" not in parameters
            or "max_k" not in parameters
            or "min_p_in" not in parameters
            or "max_p_in" not in parameters
            or "min_p_out" not in parameters
            or "max_p_out" not in parameters
        ):
            return False
        try:
            (
                number_of_graphs,
                min_n,
                max_n,
                min_k,
                max_k,
                min_p_in,
                max_p_in,
                min_p_out,
                max_p_out,
            ) = self.parse_parameters(parameters)
        except ValueError:
            return False
        if number_of_graphs < 1:
            return False
        if min_n < 0 or max_n < min_n:
            return False
        if min_k < 1 or max_k < min_k or min_k > max_n:
            return False
        if min_p_in < 0 or max_p_in < min_p_in:
            return False
        if min_p_out < 0 or max_p_out < min_p_out:
            return False
        return True

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        (
            number_of_graphs,
            min_n,
            max_n,
            min_k,
            max_k,
            min_p_in,
            max_p_in,
            min_p_out,
            max_p_out,
        ) = self.parse_parameters(parameters)
        with dataset.writer() as writer:
            for _ in range(number_of_graphs):
                n = random.randint(min_n, max_n)
                k = random.randint(min_k, min(max_k, n))
                p_in = random.uniform(min_p_in, max_p_in)
                p_out = random.uniform(min_p_out, max_p_out)
                sizes = [n // k] * k
                for i in range(n % k):
                    sizes[i] += 1
                probs = [
                    [p_in if i == j else p_out for j in range(k)] for i in range(k)
                ]
                G = nx.stochastic_block_model(sizes, probs)
                graph = create_in_memory_graph(G)
                writer.add(graph)
        return dataset
