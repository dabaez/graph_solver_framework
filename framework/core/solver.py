from typing import Protocol, runtime_checkable

from framework.core.graph import Dataset, FrameworkGraph, MaximumIndependentSet


@runtime_checkable
class DatasetSolver(Protocol):
    """Base class for solvers."""

    def solve(self, dataset: Dataset) -> list[MaximumIndependentSet]:
        """
        Solve the problem using the given dataset.

        :param dataset: The dataset to solve.
        :return: A list of MaximumIndependentSet solutions for each graph in the dataset.
        """
        ...


@runtime_checkable
class GraphSolver(Protocol):
    """Base class for graph solvers."""

    def solve_graph(self, graph: FrameworkGraph) -> MaximumIndependentSet:
        """
        Solve the problem using the given graph.

        :param graph: The graph to solve.
        :return: A MaximumIndependentSet solution for the graph.
        """
        ...


class DatasetSolverFromGraphSolver:
    """
    A solver that uses a GraphSolver to solve each graph in a dataset.
    """

    def __init__(self, graph_solver: GraphSolver):
        """
        Initializes the solver with a GraphSolver instance.

        :param graph_solver: An instance of GraphSolver.
        """
        self.graph_solver = graph_solver

    def solve(self, dataset: Dataset) -> list[MaximumIndependentSet]:
        """
        Solve each graph in the dataset.

        :param dataset: The dataset to solve.
        :return: A list of MaximumIndependentSet solutions for each graph in the dataset.
        """
        return [self.graph_solver.solve_graph(graph) for graph in dataset]
