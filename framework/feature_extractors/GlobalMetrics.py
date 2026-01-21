import math

import networkx as nx

from framework.core.feature_extractor import Feature
from framework.core.registries import register_feature_extractor
from framework.solvers.cpp_greedy.GreedyCPPSolver import GreedyCPPSolver


@register_feature_extractor("Logarithm of Nodes and Edges")
class LogarithmOfNodesAndEdges:
    def description(self) -> str:
        return "Extracts the logarithm (base 10) of the number of nodes and edges in the graph."

    def feature_names(self) -> list[str]:
        return ["log_number_of_nodes", "log_number_of_edges"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        log_num_nodes = Feature(
            name="log_number_of_nodes",
            value=math.log10(num_nodes) if num_nodes > 0 else 0,
        )
        log_num_edges = Feature(
            name="log_number_of_edges",
            value=math.log10(num_edges) if num_edges > 0 else 0,
        )

        return [log_num_nodes, log_num_edges]


@register_feature_extractor("Graph Conectivity")
class GraphConectivity:
    def description(self) -> str:
        return "Extracts whether the graph is connected and the number of connected components."

    def feature_names(self) -> list[str]:
        return ["is_connected"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        return [
            Feature(
                name="is_connected",
                value=1 if nx.is_connected(graph) else 0,
            )
        ]


@register_feature_extractor("Chromatic Number")
class ChromaticNumber:
    def description(self) -> str:
        return "Extracts the chromatic number of the graph normalized by the number of nodes."

    def feature_names(self) -> list[str]:
        return ["chromatic_number"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
        chromatic_number = max(coloring.values()) + 1
        return [
            Feature(
                name="chromatic_number",
                value=chromatic_number / graph.number_of_nodes()
                if graph.number_of_nodes() > 0
                else 0,
            )
        ]


@register_feature_extractor("Graph Assortativity")
class GraphAssortativity:
    def description(self) -> str:
        return "Extracts the assortativity coefficient of the graph."

    def feature_names(self) -> list[str]:
        return ["assortativity_coefficient"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        return [
            Feature(
                name="assortativity_coefficient",
                value=nx.degree_assortativity_coefficient(graph),
            )
        ]


@register_feature_extractor("Approximate MIS")
class ApproximateMIS:
    def description(self) -> str:
        return "Extracts the size of an approximate maximum independent set (MIS) of the graph normalized by the number of nodes."

    def feature_names(self) -> list[str]:
        return ["approximate_mis_size"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        solution = GreedyCPPSolver().solve(graph)
        return [
            Feature(
                name="approximate_mis_size",
                value=len(solution.mis) / graph.number_of_nodes()
                if graph.number_of_nodes() > 0
                else 0,
            )
        ]


@register_feature_extractor("Laplacian Eigenvalues")
class LaplacianEigenvalues:
    def description(self) -> str:
        return "Extracts the logarithm (base 10) of the largest and second smallest eigenvalues of the graph Laplacian normalized by the average node degree, and the logarithm of the eigenvalue ratio."

    def feature_names(self) -> list[str]:
        return [
            "log_largest_eigenvalue",
            "log_second_smallest_eigenvalue",
            "log_eigenvalue_ratio",
        ]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        eigenvalues = nx.laplacian_spectrum(graph)
        n = len(eigenvalues)
        ev1 = eigenvalues[-1] if n > 0 else 0
        ev2 = eigenvalues[-2] if n > 1 else 0

        avg_degree = (
            sum(dict(graph.degree()).values())  # type: ignore
            / graph.number_of_nodes()
        )

        EPS = 1e-10

        def safe_log10(value: float) -> float:
            return math.log10(value) if value > EPS else 0.0

        log_largest_eigenvalue = Feature(
            name="log_largest_eigenvalue",
            value=safe_log10(ev1 / avg_degree) if avg_degree > 0 else 0,
        )
        log_second_smallest_eigenvalue = Feature(
            name="log_second_smallest_eigenvalue",
            value=safe_log10(ev2 / avg_degree) if avg_degree > 0 else 0,
        )
        log_eigenvalue_ratio = Feature(
            name="log_eigenvalue_ratio",
            value=safe_log10(ev1 / ev2) if ev2 > 0 else 0,
        )
        return [
            log_largest_eigenvalue,
            log_second_smallest_eigenvalue,
            log_eigenvalue_ratio,
        ]
