import random

from framework.core.factories import DatasetSolverFromGraphSolver
from framework.core.graph import FrameworkGraph


class TestGraphSolver:
    def solve_graph(self, graph: FrameworkGraph) -> list[str]:
        valid_nodes = set(graph.graph_object)
        response = []
        while valid_nodes:
            node = random.choice(list(valid_nodes))
            valid_nodes.remove(node)
            response.append(node)
            neighbors = set(graph.graph_object.neighbors(node))
            valid_nodes -= neighbors
        return response


TestDatasetSolver = DatasetSolverFromGraphSolver(TestGraphSolver, "test_graph_solver")
