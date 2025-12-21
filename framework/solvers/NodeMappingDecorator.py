from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import networkx as nx

from framework.core.solver import MaximumIndependentSet, Solution


@dataclass
class IntSolution:
    mis: list[int]
    time: float


def normalize_labels(
    start_index: int,
) -> Callable[[Callable[..., IntSolution]], Callable[..., Solution]]:
    """
    Decorator to normalize node labels of a graph to a contiguous range starting from start_index.

    :param start_index: The starting index for normalization (usually 0 or 1).
    :return: A decorator that normalizes the graph's node labels.
    """

    def decorator(
        func: Callable[..., IntSolution],
    ) -> Callable[..., Solution]:
        @wraps(func)
        def wrapper(self: Any, graph: nx.Graph) -> Solution:
            original_labels = list(graph.nodes())
            label_to_normalized = {
                label: i + start_index for i, label in enumerate(original_labels)
            }
            normalized_to_label = {
                i + start_index: str(label) for i, label in enumerate(original_labels)
            }

            normalized_graph = nx.relabel_nodes(graph, label_to_normalized)

            result = func(self, normalized_graph)

            final_result = Solution(
                mis=MaximumIndependentSet(
                    [normalized_to_label[node] for node in result.mis]
                ),
                time=result.time,
            )

            return final_result

        return wrapper

    return decorator
