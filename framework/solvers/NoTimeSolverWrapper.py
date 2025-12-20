import time
from typing import Protocol, Type

import networkx as nx

from framework.core.registries import register_solver
from framework.core.solver import MaximumIndependentSet, Solution, Solver


class NoTimeSolver(Protocol):
    """Base class for solvers that do not return time."""

    def description(self) -> str:
        """
        Description of the solver.

        :return: A string describing the solver.
        """
        ...

    def solve(self, graph: nx.Graph) -> MaximumIndependentSet:
        """
        Solve the problem using the given dataset.

        :param graph: The graph to solve.
        :return: A maximum independent set of the graph.
        """
        ...


def solver_from_no_time_solver(solver: NoTimeSolver, name: str) -> Type[Solver]:
    """
    Convert a NoTimeSolver to a Solver.

    :param solver: An instance of NoTimeSolver.
    :return: A type that implements the Solver protocol.
    """

    @register_solver(name)
    class SolverImplementation:
        def description(self) -> str:
            return solver.description()

        def solve(self, graph: nx.Graph) -> Solution:
            start_time = time.time()
            mis = solver.solve(graph)
            end_time = time.time()
            return Solution(mis=MaximumIndependentSet(mis), time=end_time - start_time)

    return SolverImplementation
