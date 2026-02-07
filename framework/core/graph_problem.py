from typing import Generic, Protocol, TypeVar, runtime_checkable

from networkx import Graph


@runtime_checkable
class Solution(Protocol):
    """Base class for solutions."""

    def __dict__(self) -> dict[str, str]:
        """
        Convert the solution to a dictionary for easier comparison and storage.

        :return: A dictionary representation of the solution.
        """
        ...


SolutionT = TypeVar("SolutionT", bound=Solution, contravariant=True)


@runtime_checkable
class GraphProblem(Protocol, Generic[SolutionT]):
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

    def is_valid(self, graph: Graph, solution: SolutionT) -> bool:
        """
        Check if the given solution is valid for the graph.

        :param graph: The graph to check against.
        :param solution: The solution to validate.
        :return: True if the solution is valid, False otherwise.
        """
        ...

    def is_solution_worse(self, solution_a: SolutionT, solution_b: SolutionT) -> bool:
        """
        Compare two solutions for the problem.

        :param solution_a: The first solution to compare.
        :param solution_b: The second solution to compare.
        :return: True if solution_a is worse than solution_b, False otherwise.
        """
        ...
