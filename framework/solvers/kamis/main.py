import os
import subprocess
import tempfile
import time
from pathlib import Path

import networkx as nx

from framework.core.graph import FrameworkGraph
from framework.core.registries import register_solver
from framework.core.solver import MaximumIndependentSet, Solution
from framework.solvers.kamis.config import time_limit
from framework.solvers.NodeMappingDecorator import normalize_labels


def kamis_solver(file: str, name: str, description: str):
    @register_solver(name)
    class KamisSolver:
        def description(self) -> str:
            return description

        @normalize_labels(start_index=1)
        def solve(self, graph: FrameworkGraph) -> Solution:
            with tempfile.NamedTemporaryFile(delete=False) as input_file:
                graph_object = graph.graph_object
                with open(input_file.name, "w") as f:
                    f.write(
                        f"{graph_object.number_of_nodes()} {graph_object.number_of_edges()}\n"
                    )
                    dict = nx.to_dict_of_lists(graph_object)
                    for i in range(graph_object.number_of_nodes()):
                        neighbors = sorted(dict.get(str(i + 1), []), key=int)
                        f.write(" ".join(neighbors) + "\n")

                input_file_path = input_file.name

            output_file_path = input_file_path + ".out"
            script_path = Path(__file__).parent / "KaMIS" / "deploy" / file

            start_time = time.time()
            subprocess.run(
                f"{str(script_path)} {input_file_path} --output={output_file_path} --time_limit={time_limit}",
                shell=True,
                check=True,
                capture_output=True,
            )
            end_time = time.time()

            mis_list = []
            with open(output_file_path, "r") as output_file:
                for i, line in enumerate(output_file):
                    if line == "1":
                        mis_list.append(str(i + 1))

            os.remove(input_file_path)
            os.remove(output_file_path)

            return Solution(
                mis=MaximumIndependentSet(mis_list), time=end_time - start_time
            )

    return KamisSolver


ReduMISSolver = kamis_solver(
    "redumis",
    "ReduMISSolver",
    "Evolutionary algorithm based on graph partitioning and reduction techniques. From KaMIS.",
)

OnlineMISSolver = kamis_solver(
    "online_mis",
    "OnlineMISSolver",
    "Local search algorithm that uses (online) reductions to speed up local search. From KaMIS.",
)

WeightedBranchAndReduceSolver = kamis_solver(
    "weighted_branch_reduce",
    "WeightedBranchAndReduceSolver",
    "Branch and reduce algorithm for the weighted independent set problem. From KaMIS.",
)

WeightedLocalSearchSolver = kamis_solver(
    "weighted_local_search",
    "WeightedLocalSearchSolver",
    "Local search algorithm for the weighted independent set problem. From KaMIS.",
)

# These solver do not write to output right now, so they are disabled.
#
# MMWISSolver = kamis_solver(
#     "mmwis",
#     "Memetic Maximum Weight Independent Set Solver",
#     "Iterative reduce and evolution algorithm to solve the maximum weight independent set problem. From KaMIS.",
# )

# StructionSolver = kamis_solver(
#     "struction",
#     "StructionSolver",
#     "new branch and reduce algorithm using increasing transformations. From KaMIS.",
# )
