from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from framework.core.graph import Dataset


@dataclass
class RequiredParameter:
    name: str
    description: str
    isPath: bool = False


@runtime_checkable
class GraphCreator(Protocol):
    """Base class for graph creators."""

    def description(self) -> str:
        """
        Description of the graph creator.
        """
        ...

    def required_parameters(self) -> list[RequiredParameter]:
        """
        List of required parameters for the creator.
        """
        ...

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        """
        Validate the provided parameters against the required parameters.

        :param parameters: A dictionary of parameters to validate.
        :return: True if the parameters are valid, False otherwise.
        """
        ...

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        """
        Create graphs based on the given parameters and add them to the dataset

        :param parameters: A dictionary of parameters to create the graphs.
        :param dataset: A dataset to which to add the created graphs.

        :return: The updated dataset with the created graphs.
        """
        ...
