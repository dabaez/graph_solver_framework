from typing import Generic, Protocol, TypeVar, runtime_checkable

from networkx import Graph

T_Solution = TypeVar("T_Solution", contravariant=True)


@runtime_checkable
class GraphProblem(Protocol, Generic[T_Solution]):
    """Base class for graph problems."""

    def name(self) -> str:
        """
        Name of the graph problem.
        """
        ...

    def description(self) -> str:
        """
        Description of the graph problem.
        """
        ...

    def is_valid(self, graph: Graph, solution: T_Solution) -> bool:
        """
        Check if the given solution is valid for the graph.

        :param graph: The graph to check against.
        :param solution: The solution to validate.
        :return: True if the solution is valid, False otherwise.
        """
        ...

    def is_solution_worse(self, solution_a: T_Solution, solution_b: T_Solution) -> bool:
        """
        Compare two solutions for the problem.

        :param solution_a: The first solution to compare.
        :param solution_b: The second solution to compare.
        :return: True if solution_a is worse than solution_b, False otherwise.
        """
        ...
