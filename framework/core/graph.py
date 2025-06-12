from networkx import Graph as NetworkXGraph


class Graph:
    """
    A base class for a graph structure.
    """

    graph_object: NetworkXGraph
    features: dict

    def __init__(self, graph_object: NetworkXGraph, features: dict | None = None):
        """
        Initializes the Graph with a NetworkX graph object and optional features.

        :param graph_object: A NetworkX graph object.
        :param features: Optional dictionary of features associated with the graph.
        """
        self.graph_object = graph_object
        self.features = features if features is not None else {}


class Dataset:
    """
    A base class for a dataset containing multiple graphs.
    """

    name: str
    graphs: list[Graph]

    def __init__(self, name: str | None, graphs: list[Graph]):
        """
        Initializes the Dataset with a name and a list of Graph objects.

        :param name: The name of the dataset.
        :param graphs: A list of Graph objects.
        """
        if name is None:
            self.name = "Unnamed Dataset"
        else:
            self.name = name
        self.graphs = graphs

    def add_graph(self, graph: Graph):
        """
        Adds a Graph object to the dataset.

        :param graph: A Graph object to add to the dataset.
        """
        self.graphs.append(graph)

    def __len__(self) -> int:
        """
        Returns the number of graphs in the dataset.

        :return: The number of graphs in the dataset.
        """
        return len(self.graphs)

    def __getitem__(self, index: int) -> Graph:
        """
        Returns the Graph object at the specified index.

        :param index: The index of the graph to retrieve.
        :return: The Graph object at the specified index.
        """
        return self.graphs[index]

    def __iter__(self):
        """
        Returns an iterator over the graphs in the dataset.

        :return: An iterator over the Graph objects in the dataset.
        """
        return iter(self.graphs)
