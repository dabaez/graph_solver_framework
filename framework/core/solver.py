from typing import Generic, NewType, Protocol, TypeVar, runtime_checkable

from networkx import Graph

MaximumIndependentSet = NewType("MaximumIndependentSet", list[str])

T_Solution = TypeVar("T_Solution", covariant=True)


@runtime_checkable
class Solver(Protocol, Generic[T_Solution]):
    """Base class for solvers."""

    def description(self) -> str:
        """
        Description of the solver.

        :return: A string describing the solver.
        """
        ...

    def solve(self, graph: Graph) -> T_Solution:
        """
        Solve the problem using the given dataset.

        :param graph: The graph to solve.
        :return: A maximum independent set of the graph.
        """
        ...
