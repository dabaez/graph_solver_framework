import numpy as np

from framework.core.graph import Feature, FrameworkGraph
from framework.core.registries import register_feature_extractor
from framework.solvers.cpp_greedy.GreedyCPPSolver import GreedyCPPSolver


@register_feature_extractor("random_weights")
class RandomWeights:
    def description(self) -> str:
        return (
            "Adds a random weight to every node from each graph for Machine Learning."
        )

    def feature_names(self) -> list[str]:
        return ["weights"]

    def extract_features(self, graph: FrameworkGraph) -> list[Feature]:
        return [
            Feature(
                name="weights",
                value=np.around(
                    np.random.normal(1, 0.1, graph.graph_object.number_of_nodes())
                    .astype(int)
                    .clip(min=0)
                ),
            )
        ]


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
