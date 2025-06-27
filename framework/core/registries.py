from typing import Callable, Type

from framework.core.dataset_creator import DatasetCreator
from framework.core.feature_extractor import DatasetFeatureExtractor
from framework.core.solver import DatasetSolver

DATASET_CREATORS: dict[str, Type[DatasetCreator]] = {}
DATASET_SOLVERS: dict[str, Type[DatasetSolver]] = {}
DATASET_FEATURE_EXTRACTORS: dict[str, Type[DatasetFeatureExtractor]] = {}


def register_dataset_creator(
    name: str,
) -> Callable[[Type[DatasetCreator]], Type[DatasetCreator]]:
    """
    Decorator to register a dataset creator.

    :param name: The name of the dataset creator.
    :return: A decorator that registers the dataset creator.
    """

    def decorator(creator: Type[DatasetCreator]) -> Type[DatasetCreator]:
        if name in DATASET_CREATORS:
            raise ValueError(
                f"Dataset creator with name '{name}' is already registered."
            )
        DATASET_CREATORS[name] = creator
        return creator

    return decorator


def register_dataset_solver(
    name: str,
) -> Callable[[Type[DatasetSolver]], Type[DatasetSolver]]:
    """
    Decorator to register a dataset solver.

    :param name: The name of the dataset solver.
    :return: A decorator that registers the dataset solver.
    """

    def decorator(solver: Type[DatasetSolver]) -> Type[DatasetSolver]:
        if name in DATASET_SOLVERS:
            raise ValueError(
                f"Dataset solver with name '{name}' is already registered."
            )
        DATASET_SOLVERS[name] = solver
        return solver

    return decorator


def register_dataset_feature_extractor(
    name: str,
) -> Callable[[Type[DatasetFeatureExtractor]], Type[DatasetFeatureExtractor]]:
    """
    Decorator to register a dataset feature extractor.

    :param name: The name of the dataset feature extractor.
    :return: A decorator that registers the dataset feature extractor.
    """

    def decorator(
        extractor: Type[DatasetFeatureExtractor],
    ) -> Type[DatasetFeatureExtractor]:
        if name in DATASET_FEATURE_EXTRACTORS:
            raise ValueError(
                f"Dataset feature extractor with name '{name}' is already registered."
            )
        DATASET_FEATURE_EXTRACTORS[name] = extractor
        return extractor

    return decorator
