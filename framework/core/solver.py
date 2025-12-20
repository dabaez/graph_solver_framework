from dataclasses import dataclass
from typing import NewType, Protocol, runtime_checkable

from networkx import Graph

MaximumIndependentSet = NewType("MaximumIndependentSet", list[str])


@dataclass
class Solution:
    mis: MaximumIndependentSet
    time: float


@runtime_checkable
class Solver(Protocol):
    """Base class for solvers."""

    def description(self) -> str:
        """
        Description of the solver.

        :return: A string describing the solver.
        """
        ...

    def solve(self, graph: Graph) -> Solution:
        """
        Solve the problem using the given dataset.

        :param graph: The graph to solve.
        :return: A maximum independent set of the graph.
        """
        ...
