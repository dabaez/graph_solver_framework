from typing import Callable, Type

from framework.core.feature_extractor import FeatureExtractor
from framework.core.graph_creator import GraphCreator
from framework.core.graph_problem import GraphProblem
from framework.core.solver import Solver

GRAPH_CREATORS: dict[str, Type[GraphCreator]] = {}
SOLVERS: dict[str, dict[str, Type[Solver]]] = {}
FEATURE_EXTRACTORS: dict[str, Type[FeatureExtractor]] = {}
PROBLEMS: dict[str, Type[GraphProblem]] = {}


def register_graph_creator(
    name: str,
) -> Callable[[Type[GraphCreator]], Type[GraphCreator]]:
    """
    Decorator to register a dataset creator.

    :param name: The name of the dataset creator.
    :return: A decorator that registers the dataset creator.
    """

    def decorator(creator: Type[GraphCreator]) -> Type[GraphCreator]:
        if name in GRAPH_CREATORS:
            raise ValueError(
                f"Dataset creator with name '{name}' is already registered."
            )
        GRAPH_CREATORS[name] = creator
        return creator

    return decorator


def register_solver(
    problem_name: str,
    name: str,
) -> Callable[[Type[Solver]], Type[Solver]]:
    """
    Decorator to register a solver.

    :param name: The name of the solver.
    :return: A decorator that registers the solver.
    """

    def decorator(solver: Type[Solver]) -> Type[Solver]:
        if problem_name in SOLVERS and name in SOLVERS[problem_name]:
            raise ValueError(f"Solver with name '{name}' is already registered.")
        if problem_name not in SOLVERS:
            SOLVERS[problem_name] = {}
        SOLVERS[problem_name][name] = solver
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


def register_problem(
    name: str,
) -> Callable[[Type[GraphProblem]], Type[GraphProblem]]:
    """
    Decorator to register a graph problem.

    :param name: The name of the graph problem.
    :return: A decorator that registers the graph problem.
    """

    def decorator(
        problem: Type[GraphProblem],
    ) -> Type[GraphProblem]:
        if name in PROBLEMS:
            raise ValueError(f"Graph problem with name '{name}' is already registered.")
        PROBLEMS[name] = problem
        return problem

    return decorator
