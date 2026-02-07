import random

import networkx as nx

from problems.maximum_independent_set.shared.NoTimeSolverWrapper import (
    solver_from_no_time_solver,
)


class TestGraphSolver:
    def description(self) -> str:
        return "A test solver that randomly selects nodes to form a maximum independent set."

    def solve(self, graph: nx.Graph) -> list[str]:
        valid_nodes = set(graph)
        response = []
        while valid_nodes:
            node = random.choice(list(valid_nodes))
            valid_nodes.remove(node)
            response.append(str(node))
            neighbors = set(graph.neighbors(node))
            valid_nodes -= neighbors
        return response


TestGraphSolverImplementation = solver_from_no_time_solver(
    TestGraphSolver(), "MaximumIndependentSetProblem", "TestGraphSolver"
)
