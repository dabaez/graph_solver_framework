from typing import Callable, Type

from framework.core.feature_extractor import FeatureExtractor
from framework.core.graph_creator import GraphCreator
from framework.core.solver import Solver

DATASET_CREATORS: dict[str, Type[GraphCreator]] = {}
SOLVERS: dict[str, Type[Solver]] = {}
FEATURE_EXTRACTORS: dict[str, Type[FeatureExtractor]] = {}


def register_dataset_creator(
    name: str,
) -> Callable[[Type[GraphCreator]], Type[GraphCreator]]:
    """
    Decorator to register a dataset creator.

    :param name: The name of the dataset creator.
    :return: A decorator that registers the dataset creator.
    """

    def decorator(creator: Type[GraphCreator]) -> Type[GraphCreator]:
        if name in DATASET_CREATORS:
            raise ValueError(
                f"Dataset creator with name '{name}' is already registered."
            )
        DATASET_CREATORS[name] = creator
        return creator

    return decorator


def register_solver(
    name: str,
) -> Callable[[Type[Solver]], Type[Solver]]:
    """
    Decorator to register a solver.

    :param name: The name of the solver.
    :return: A decorator that registers the solver.
    """

    def decorator(solver: Type[Solver]) -> Type[Solver]:
        if name in SOLVERS:
            raise ValueError(f"Solver with name '{name}' is already registered.")
        SOLVERS[name] = solver
        return solver

    return decorator


def register_feature_extractor(
    name: str,
) -> Callable[[Type[FeatureExtractor]], Type[FeatureExtractor]]:
    """
    Decorator to register a dataset feature extractor.

    :param name: The name of the feature extractor.
    :return: A decorator that registers the feature extractor.
    """

    def decorator(
        extractor: Type[FeatureExtractor],
    ) -> Type[FeatureExtractor]:
        if name in FEATURE_EXTRACTORS:
            raise ValueError(
                f"Feature extractor with name '{name}' is already registered."
            )
        FEATURE_EXTRACTORS[name] = extractor
        return extractor

    return decorator
