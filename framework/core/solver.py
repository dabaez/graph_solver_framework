from typing import Protocol, runtime_checkable

from framework.core.graph import Dataset, MaximumIndependentSet


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
