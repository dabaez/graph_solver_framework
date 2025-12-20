from functools import wraps
from typing import Any, Callable

import networkx as nx

from framework.core.graph import FrameworkGraph
from framework.core.solver import MaximumIndependentSet, Solution


def normalize_labels(start_index: int) -> Callable:
    """
    Decorator to normalize node labels of a graph to a contiguous range starting from start_index.

    :param start_index: The starting index for normalization (usually 0 or 1).
    :return: A decorator that normalizes the graph's node labels.
    """

    def decorator(func: Callable[[Any, nx.Graph], Solution]) -> Callable:
        @wraps(func)
        def wrapper(self, graph: nx.Graph) -> Solution:
            original_labels = list(graph.nodes())
            label_to_normalized = {
                label: i + start_index for i, label in enumerate(original_labels)
            }
            normalized_to_label = {
                i + start_index: str(label) for i, label in enumerate(original_labels)
            }

            normalized_graph = nx.relabel_nodes(graph, label_to_normalized)
            normalized_framework_graph = FrameworkGraph(normalized_graph)

            result = func(self, normalized_framework_graph)

            result.mis = MaximumIndependentSet(
                [normalized_to_label[node] for node in result.mis]
            )

            return result

        return wrapper

    return decorator
