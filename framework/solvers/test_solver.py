import random

from framework.core.factories import DatasetSolverFromGraphSolver
from framework.core.graph import FrameworkGraph, MaximumIndependentSet


class TestGraphSolver:
    def solve_graph(self, graph: FrameworkGraph) -> MaximumIndependentSet:
        valid_nodes = set(graph.graph_object)
        response = []
        while valid_nodes:
            node = random.choice(list(valid_nodes))
            valid_nodes.remove(node)
            response.append(node)
            neighbors = set(graph.graph_object.neighbors(node))
            valid_nodes -= neighbors
        return MaximumIndependentSet(response)


TestDatasetSolver = DatasetSolverFromGraphSolver(TestGraphSolver, "test_graph_solver")
