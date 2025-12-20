import os

import networkx as nx

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_dataset_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_dataset_creator("DIMACStoMISLoader")
class DIMACStoMISLoader:
    def description(self) -> str:
        return "Loads a dataset of DIMACS Maximum Clique instances and converts them to Maximum Independent Set (MIS) instances."

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="folder path",
                description="Path to the folder containing the DIMACS Maximum Clique instance files.",
                isPath=True,
            ),
        ]

    def validate_parameters(self, parameters: dict[str, str]) -> bool:
        if "folder path" not in parameters:
            return False
        folder_path = parameters["folder path"]
        return os.path.isdir(folder_path)

    def create_graphs(self, parameters: dict[str, str], dataset: Dataset) -> Dataset:
        folder_path = parameters["folder path"]

        for filename in os.listdir(folder_path):
            if filename.endswith(".clq"):
                file_path = os.path.join(folder_path, filename)
                graph = self.convert_clique_to_mis(file_path)
                framework_graph = create_in_memory_graph(
                    graph, {"source_file": filename}
                )
                dataset.append(framework_graph)

        return dataset

    def convert_clique_to_mis(self, file_path: str) -> nx.Graph:
        """Convert a DIMACS Maximum Clique instance to a Maximum Independent Set (MIS) instance."""
        with open(file_path, "r") as f:
            lines = f.readlines()

        edges = []
        num_vertices = 0
        for line in lines:
            if line.startswith("p"):
                parts = line.split()
                num_vertices = int(parts[2])
            elif line.startswith("e"):
                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])
                edges.append((u, v))

        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(range(1, num_vertices + 1))
        clique_graph.add_edges_from(edges)
        mis_graph = nx.complement(clique_graph)

        return mis_graph
