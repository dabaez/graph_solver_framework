import framework.dataset_creators  # noqa: F401
import framework.feature_extractors  # noqa: F401
import framework.solvers  # noqa: F401
from framework.core.registries import SOLVERS
from framework.experiment.utils import load_dataset, save_solver_solution

dataset = load_dataset("dimacs")

solvers = [
    "ReduMISSolver",
    "OnlineMISSolver",
    "WeightedBranchAndReduceSolver",
    "WeightedLocalSearchSolver",
    "GreedyCPPSolver",
]

for solver_name in solvers:
    solver_class = SOLVERS.get(solver_name)
    if not solver_class:
        print("AAAAAAAA")
        continue

    solver_instance = solver_class()
    solutions = []
    for i, graph in enumerate(dataset):
        print(f"Solving graph {i + 1}/{len(dataset)} with {solver_name}...")
        solution = solver_instance.solve(graph)
        solutions.append(solution)
    solution_file = save_solver_solution(solver_name, solutions, "dimacs")
    print(f"Solutions saved to '{solution_file}'.")
