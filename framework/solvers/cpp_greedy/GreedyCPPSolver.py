import time

from framework.core.graph import FrameworkGraph
from framework.core.registries import register_solver
from framework.core.solver import MaximumIndependentSet, Solution
from framework.solvers.NodeMappingDecorator import normalize_labels

try:
    import framework.solvers.cpp_greedy.mis_greedy_cpp as mis_greedy_cpp
except ImportError:
    mis_greedy_cpp = None


@register_solver("GreedyCPPSolver")
class GreedyCPPSolver:
    def description(self) -> str:
        return "A solver that uses a C++ greedy algorithm to find a maximum independent set. Takes the current lowest degree node and removes it and its neighbors from the graph until no nodes are left."

    @normalize_labels(start_index=0)
    def solve(self, graph: FrameworkGraph) -> Solution:
        if mis_greedy_cpp is None:
            raise ImportError("C++ solver module is not available.")

        start_time = time.time()

        mis = mis_greedy_cpp.solve(
            graph.graph_object.number_of_nodes(), graph.graph_object.edges()
        )

        end_time = time.time()

        return Solution(mis=MaximumIndependentSet(mis), time=end_time - start_time)
