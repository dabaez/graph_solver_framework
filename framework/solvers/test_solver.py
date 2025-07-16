import random

from framework.core.graph import FrameworkGraph
from framework.core.registries import register_solver
from framework.core.solver import MaximumIndependentSet


@register_solver("test_graph_solver")
class TestGraphSolver:
    def description(self) -> str:
        return "A test solver that randomly selects nodes to form a maximum independent set."

    def solve(self, graph: FrameworkGraph) -> MaximumIndependentSet:
        valid_nodes = set(graph.graph_object)
        response = MaximumIndependentSet([])
        while valid_nodes:
            node = random.choice(list(valid_nodes))
            valid_nodes.remove(node)
            response.append(node)
            neighbors = set(graph.graph_object.neighbors(node))
            valid_nodes -= neighbors
        return response
