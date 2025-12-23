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
