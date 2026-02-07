import networkx as nx
import numpy as np

from framework.core.feature_extractor import Feature
from framework.core.registries import register_feature_extractor
from problems.maximum_independent_set.solvers.cpp_greedy.GreedyCPPSolver import (
    GreedyCPPSolver,
)


## VERY IMPORTANT TO REPLACE GREEDY LABELS WITH ACTUAL LABELS ON THE FUTURE
@register_feature_extractor("greedy_labels")
class GreedyLabels:
    def description(self) -> str:
        return "Applies a greedy algorithm to assign labels to nodes."

    def feature_names(self) -> list[str]:
        return ["labels"]

    def extract_features(self, graph: nx.Graph) -> list[Feature]:
        solution = GreedyCPPSolver().solve(graph)
        return [
            Feature(
                name="labels",
                value=np.array(
                    [int(str(node_label) in solution.set) for node_label in graph.nodes]
                ),
            )
        ]
