from networkx import Graph

from framework.core.graph import Dataset, FrameworkGraph

##### DATASET #####


def DatasetFromNetworkX(graphs: list[Graph]) -> Dataset:
    return Dataset([FrameworkGraph(graph_object=graph) for graph in graphs])
