solvers = [
    "ReduMISSolver",
    "OnlineMISSolver",
    "WeightedBranchAndReduceSolver",
    "WeightedLocalSearchSolver",
    "GreedyCPPSolver",
]

import framework.dataset_creators  # noqa: F401
import framework.feature_extractors  # noqa: F401
import framework.solvers  # noqa: F401
from framework.experiment.utils import load_dataset


def analyze_solvers_performance():
    """
    Analyze solver performance on DIMACS dataset.
    For each graph, print its characteristics and which solver did best in size and time.
    """
    # Load the DIMACS dataset
    dataset = load_dataset("dimacs")

    print(f"Loaded dataset with {len(dataset)} graphs")

    # Get all solution files for DIMACS dataset
    import os

    solutions_folder = "data/solutions"
    solution_files = [
        f for f in os.listdir(solutions_folder) if f.endswith("_dimacs.csv")
    ]

    # Map solver names to their solution files
    solver_solution_map = {}
    for file in solution_files:
        # Extract solver name from filename (e.g., GreedyCPPSolver_dimacs.csv)
        solver_name = file.replace("_dimacs.csv", "")
        solver_solution_map[solver_name] = os.path.join(solutions_folder, file)

    print(f"Found solution files for solvers: {list(solver_solution_map.keys())}")

    # For each graph, analyze which solver did best
    for i, graph in enumerate(dataset[:10]):
        print(f"\n--- Graph {i + 1} ---")

        # Print graph characteristics
        # print("Characteristics:")
        # for feature in graph.features:
        # print(f"  {feature}: {graph.features[feature]}")

        # Get the best solver for size and time for this graph
        best_size_solver = None
        best_time_solver = None
        max_size = -float("inf")
        min_time = float("inf")

        # Check each solver that has solutions
        for solver_name in solvers:
            if solver_name not in solver_solution_map:
                continue

            # Read the solution file for this solver
            try:
                import csv

                with open(solver_solution_map[solver_name], "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header
                    # Get the solution for this specific graph (row index i)
                    row = None
                    for j, r in enumerate(reader):
                        if j == i:
                            row = r
                            break

                    if row:
                        solution_length = int(row[1])  # Solution length (size)
                        time_taken = float(row[2])  # Time taken

                        print(
                            f"Solver: {solver_name}, Size: {solution_length}, Time: {time_taken}"
                        )

                        # Check for best size (minimum size)
                        if solution_length > max_size:
                            max_size = solution_length
                            best_size_solver = solver_name

                        # Check for best time (minimum time)
                        if time_taken < min_time:
                            min_time = time_taken
                            best_time_solver = solver_name
            except Exception as e:
                print(f"Error reading {solver_name} solution: {e}")
                continue

        print(f"Best solver by size: {best_size_solver} (size: {max_size})")
        print(f"Best solver by time: {best_time_solver} (time: {min_time})")


# Run the analysis
analyze_solvers_performance()
