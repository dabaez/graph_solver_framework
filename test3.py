import csv
import os

from framework.experiment.utils import load_dataset

solvers = [
    "ReduMISSolver",
    "OnlineMISSolver",
    "WeightedBranchAndReduceSolver",
    "WeightedLocalSearchSolver",
    "GreedyCPPSolver",
]

examined_solver = "GreedyCPPSolver"

solutions_folder = "data/solutions"
solution_files = [f for f in os.listdir(solutions_folder) if f.endswith("_dimacs.csv")]


solver_solution_map = {}
for file in solution_files:
    solver_name = file.replace("_dimacs.csv", "")
    solver_solution_map[solver_name] = os.path.join(solutions_folder, file)

print(f"Found solution files for solvers: {list(solver_solution_map.keys())}")

solutions = {}

for solver_name in solvers:
    with open(solver_solution_map[solver_name], "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        solver_solutions = []
        for row in reader:
            graph = int(row[0])
            size = int(row[1])
            time = float(row[2])
            solver_solutions.append((graph, size, time))
        solutions[solver_name] = solver_solutions

online_places = []

for i, graph in enumerate(solutions[examined_solver]):
    place = 1
    for solver_name in solvers:
        if solutions[solver_name][i][1] > solutions[examined_solver][i][1]:
            place += 1
    online_places.append(place)

# online place statistics
print(f"\n{examined_solver} place statistics:")
print(f"Mean place: {sum(online_places) / len(online_places):.2f}")
print(f"Min place: {min(online_places)}")
print(f"Max place: {max(online_places)}")
print(
    "Places distribution: "
    + ", ".join(f"{p}: {online_places.count(p)}" for p in range(1, len(solvers) + 1))
)

for i, place in enumerate(online_places):
    if place == 5:
        print(f"Graph {i + 1} where {examined_solver} placed 5th")


dataset = load_dataset("dimacs")

x = {}

for feature_name, feature_value in dataset[0].features.items():
    if feature_name != "source_file":
        x[feature_name] = []

for i, graph in enumerate(dataset):
    features = graph.features
    for feature_name, feature_value in features.items():
        if feature_name != "source_file":
            x[feature_name].append(feature_value)

y = online_places

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(
    [list(x[feature_name][i] for feature_name in x.keys()) for i in range(len(y))],
    y,
)

import matplotlib.pyplot as plt
from sklearn import tree

print(f"Decision Tree for {examined_solver} trained on graph features:")

plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf,
    feature_names=list(x.keys()),
    class_names=[str(i) for i in range(1, len(solvers) + 1)],
    filled=True,
)
# save to file
plt.savefig(f"{examined_solver.lower()}_decision_tree.png")
plt.show()
