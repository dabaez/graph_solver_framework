import time

from framework.core.graph import FrameworkGraph
from framework.core.registries import register_solver
from framework.core.solver import MaximumIndependentSet, Solution

try:
    import framework.solvers.cpp_greedy.mis_greedy_cpp as mis_greedy_cpp
except ImportError:
    mis_greedy_cpp = None


@register_solver("GreedyCPPSolver")
class GreedyCPPSolver:
    def description(self) -> str:
        return "A solver that uses a C++ greedy algorithm to find a maximum independent set. Takes the current lowest degree node and removes it and its neighbors from the graph until no nodes are left."

    def solve(self, graph: FrameworkGraph) -> Solution:
        if mis_greedy_cpp is None:
            raise ImportError("C++ solver module is not available.")

        node_labels = list(graph.graph_object.nodes())
        label_to_node = {label: i for i, label in enumerate(node_labels)}
        edges = [
            (label_to_node[u], label_to_node[v]) for u, v in graph.graph_object.edges()
        ]

        start_time = time.time()

        mis = mis_greedy_cpp.solve(len(node_labels), edges)

        end_time = time.time()

        LabeledMIS = [str(node_labels[i]) for i in mis]

        return Solution(
            mis=MaximumIndependentSet(LabeledMIS), time=end_time - start_time
        )
