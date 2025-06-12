from typing import Protocol

from framework.core.graph import Dataset, Graph


class DatasetSolver(Protocol):
    """
    Base class for solvers.
    """

    def solve(self, dataset: Dataset) -> list[dict]:
        """
        Solve the problem using the given dataset.

        :param dataset: The dataset to solve.
        :return: A new dataset with the solution applied.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GraphSolver(Protocol):
    """
    Base class for graph solvers.
    """

    def solve_graph(self, graph: Graph) -> dict:
        """
        Solve the problem using the given graph.

        :param graph: The graph to solve.
        :return: A dictionary containing the solution.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class DatasetSolverFromGraphSolver:
    """
    A solver that uses a GraphSolver to solve graphs and create a solution for the dataset.
    """

    def __init__(self, graph_solver: GraphSolver):
        """
        Initializes the solver with a GraphSolver instance.

        :param graph_solver: An instance of GraphSolver.
        """
        self.graph_solver = graph_solver

    def solve(self, dataset: Dataset) -> list[dict]:
        """
        Solve the dataset by solving each graph in the dataset.

        :param dataset: The dataset to solve.
        :return: A dictionary containing the solutions for each graph in the dataset.
        """

        solutions = []
        for graph in dataset:
            solution = self.graph_solver.solve_graph(graph)
            solutions.append(solution)
        return solutions
