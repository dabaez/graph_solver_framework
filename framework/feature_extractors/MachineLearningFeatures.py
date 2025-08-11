import numpy as np

from framework.core.graph import Feature, FrameworkGraph
from framework.core.registries import register_feature_extractor
from framework.solvers.cpp_greedy.GreedyCPPSolver import GreedyCPPSolver


## VERY IMPORTANT TO REPLACE GREEDY LABELS WITH ACTUAL LABELS ON THE FUTURE
@register_feature_extractor("greedy_labels")
class GreedyLabels:
    def description(self) -> str:
        return "Applies a greedy algorithm to assign labels to nodes."

    def feature_names(self) -> list[str]:
        return ["labels"]

    def extract_features(self, graph: FrameworkGraph) -> list[Feature]:
        solution = GreedyCPPSolver().solve(graph)
        return [
            Feature(
                name="labels",
                value=np.array(
                    [
                        int(str(node_label) in solution.mis)
                        for node_label in graph.graph_object.nodes
                    ]
                ),
            )
        ]
