from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from framework.core.graph import Dataset


@dataclass
class RequiredParameter:
    name: str
    description: str


@runtime_checkable
class DatasetCreator(Protocol):
    """Base class for dataset creators."""

    def description(self) -> str:
        """
        Description of the dataset creator.
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

    def create_dataset(self, parameters: dict[str, str]) -> Dataset:
        """
        Create a dataset based on the provided parameters.

        :param parameters: A dictionary of parameters to create the dataset.
        :return: A Dataset object created from the parameters.
        """
        ...
