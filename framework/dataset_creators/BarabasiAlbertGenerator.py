import networkx as nx

from framework.core.dataset_creator import RequiredParameter
from framework.core.graph import Dataset, FrameworkGraph
from framework.core.registries import register_dataset_creator


@register_dataset_creator("BarabasiAlbertGenerator")
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
        step_n = int(parameters.get("step n", 1))
        step_m = int(parameters.get("step m", 1))
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

    def create_dataset(self, parameters: dict[str, str]) -> Dataset:
        min_n, max_n, min_m, max_m, step_n, step_m = self.parse_parameters(parameters)
        dataset = Dataset([])
        for n in range(min_n, max_n + 1, step_n):
            for m in range(min_m, max_m + 1, step_m):
                if 1 <= m < n:
                    G = nx.barabasi_albert_graph(n, m)
                    dataset.append(FrameworkGraph(G))
        return dataset
