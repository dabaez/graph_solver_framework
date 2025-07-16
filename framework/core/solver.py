from dataclasses import dataclass
from typing import NewType, Protocol, runtime_checkable

from framework.core.graph import FrameworkGraph

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

    def solve(self, graph: FrameworkGraph) -> Solution:
        """
        Solve the problem using the given dataset.

        :param graph: The graph to solve.
        :return: A maximum independent set of the graph.
        """
        ...
