import time

import networkx as nx

from framework.core.registries import register_solver
from problems.maximum_independent_set.shared.NodeMappingDecorator import (
    IntSolution,
    normalize_labels,
)

try:
    import problems.maximum_independent_set.solvers.cpp_greedy.mis_greedy_cpp as mis_greedy_cpp
except ImportError:
    mis_greedy_cpp = None


@register_solver("MaximumIndependentSetProblem", "GreedyCPPSolver")
class GreedyCPPSolver:
    def description(self) -> str:
        return "A solver that uses a C++ greedy algorithm to find a maximum independent set. Takes the current lowest degree node and removes it and its neighbors from the graph until no nodes are left."

    @normalize_labels(start_index=0)
    def solve(self, graph: nx.Graph) -> IntSolution:
        if mis_greedy_cpp is None:
            raise ImportError("C++ solver module is not available.")

        start_time = time.time()

        mis = mis_greedy_cpp.solve(graph.number_of_nodes(), graph.edges())

        end_time = time.time()

        return IntSolution(mis=mis, time=end_time - start_time)
