import os

import networkx as nx
import numpy as np
from pysat.formula import CNF

from framework.core.graph import Dataset
from framework.core.graph_creator import RequiredParameter
from framework.core.registries import register_graph_creator
from framework.dataset.MemoryDataset import create_in_memory_graph


@register_graph_creator("SATtoMISLoader")
class SATtoMISLoader:
    def description(self) -> str:
        return "Loads a dataset of SAT instances and converts them to Maximum Independent Set (MIS) instances."

    def required_parameters(self) -> list[RequiredParameter]:
        return [
            RequiredParameter(
                name="folder path",
                description="Path to the folder containing the SAT instance files. The instances should be in DIMACS cnf format.",
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

        with dataset.writer() as writer:
            for filename in os.listdir(folder_path):
                if filename.endswith(".cnf"):
                    file_path = os.path.join(folder_path, filename)
                    graph = self.convert_sat_to_mis(file_path)
                    framework_graph = create_in_memory_graph(
                        graph, metadata={"source_file": filename}
                    )
                    writer.add(framework_graph)

        return dataset

    # taken from mis benchmark framework https://github.com/MaxiBoether/mis-benchmark-framework/blob/master/data_generation/sat.py
    def convert_sat_to_mis(self, file_path: str) -> nx.Graph:
        """Convert a SAT instance in DIMACS cnf format to a Maximum Independent Set (MIS) instance."""
        cnf = CNF(from_file=file_path)
        nv = cnf.nv
        clauses = list(filter(lambda x: x, cnf.clauses))
        ind = {
            k: [] for k in np.concatenate([np.arange(1, nv + 1), -np.arange(1, nv + 1)])
        }
        edges = []
        for i, clause in enumerate(clauses):
            a = clause[0]
            b = clause[1]
            c = clause[2]
            aa = 3 * i
            bb = 3 * i + 1
            cc = 3 * i + 2
            ind[a].append(aa)
            ind[b].append(bb)
            ind[c].append(cc)
            edges.append((aa, bb))
            edges.append((aa, cc))
            edges.append((bb, cc))

        for i in range(1, nv + 1):
            for u in ind[i]:
                for v in ind[-i]:
                    edges.append((u, v))

        return nx.from_edgelist(edges)
