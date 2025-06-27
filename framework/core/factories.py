from typing import Protocol, Type, runtime_checkable

from networkx import Graph

from framework.core.feature_extractor import DatasetFeatureExtractor
from framework.core.graph import Dataset, Feature, FrameworkGraph, MaximumIndependentSet
from framework.core.registries import (
    register_dataset_feature_extractor,
    register_dataset_solver,
)
from framework.core.solver import DatasetSolver

##### DATASET #####


def DatasetFromNetworkX(graphs: list[Graph]) -> Dataset:
    return Dataset([FrameworkGraph(graph_object=graph) for graph in graphs])


##### FEATURE EXTRACTOR #####


@runtime_checkable
class GraphFeatureExtractor(Protocol):
    """Base class for graph feature extractors."""

    def extract_features(self, graph: FrameworkGraph) -> list[Feature]:
        """
        Extract features from the graph.

        :param graph: The graph from which to extract features.
        :return: A list of features extracted from the graph.
        """
        ...


def DatasetFeatureExtractorFromGraphFeatureExtractor(
    GraphFeatureExtractor: GraphFeatureExtractor, name: str
) -> Type[DatasetFeatureExtractor]:
    """
    A feature extractor that uses a GraphFeatureExtractor to extract features from each graph in a dataset.
    """

    @register_dataset_feature_extractor(name)
    class DatasetFeatureExtractorImpl:
        def extract_features(self, dataset: Dataset) -> list[list[Feature]]:
            """
            Extract features from each graph in the dataset.

            :param dataset: The dataset from which to extract features.
            :return: A list of lists of features extracted from each graph in the dataset.
            """
            return [GraphFeatureExtractor.extract_features(graph) for graph in dataset]

    return DatasetFeatureExtractorImpl


##### SOLVER ####


@runtime_checkable
class GraphSolver(Protocol):
    """Base class for graph solvers."""

    def solve_graph(self, graph: FrameworkGraph) -> MaximumIndependentSet:
        """
        Solve the problem using the given graph.

        :param graph: The graph to solve.
        :return: A MaximumIndependentSet solution for the graph.
        """
        ...


def DatasetSolverFromGraphSolver(
    GraphSolver: GraphSolver, name: str
) -> Type[DatasetSolver]:
    """
    A solver that uses a GraphSolver to solve each graph in a dataset.
    """

    @register_dataset_solver(name)
    class DatasetSolverImpl:
        def solve(self, dataset: Dataset) -> list[MaximumIndependentSet]:
            """
            Solve the problem using the given dataset.

            :param dataset: The dataset to solve.
            :return: A list of MaximumIndependentSet solutions for each graph in the dataset.
            """
            return [GraphSolver.solve_graph(graph) for graph in dataset]

    return DatasetSolverImpl
