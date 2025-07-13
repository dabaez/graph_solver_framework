from typing import Protocol, runtime_checkable

from framework.core.graph import Dataset
from framework.core.solution import Solution


@runtime_checkable
class DatasetSolver(Protocol):
    """Base class for solvers."""

    def solve(self, dataset: Dataset) -> list[Solution]:
        """
        Solve the problem using the given dataset.

        :param dataset: The dataset to solve.
        :return: A list of solutions for the dataset.
        """
        ...
