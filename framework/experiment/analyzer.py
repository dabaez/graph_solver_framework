import os
import sys

import matplotlib.pyplot as plt
import questionary
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from framework.core.graph import Dataset
from framework.core.registries import PROBLEMS
from framework.experiment.config import ANALYSIS_FOLDER, SOLUTIONS_FOLDER
from framework.experiment.utils import (
    fully_calculated_features,
    list_datasets,
    load_dataset,
    load_solver_solution,
)


def list_problems(dataset: str) -> list[str]:
    solutions_path = os.path.join(SOLUTIONS_FOLDER, dataset)
    problems = []
    for folder in os.listdir(solutions_path):
        if os.path.isdir(os.path.join(solutions_path, folder)):
            problems.append(folder)
    return problems


def list_solutions(dataset: str, problem_name: str) -> list[str]:
    solutions_path = os.path.join(SOLUTIONS_FOLDER, dataset, problem_name)
    solutions = []
    for file in os.listdir(solutions_path):
        if file.endswith(".csv"):
            solutions.append(file[:-4])
    return solutions


def analyzer(
    dataset_name: str,
    dataset: Dataset,
    problem_name: str,
    solvers: list[str],
    features: list[str],
):
    solutions_logs = {}
    for solver in solvers:
        solutions_logs[solver] = load_solver_solution(
            problem_name, solver, dataset_name
        )
        assert len(solutions_logs[solver]) == len(dataset), (
            f"Solution log for {solver} is not complete"
        )

    X = []
    for i, graph in enumerate(dataset):
        feature_vector = []
        for feature in features:
            feature_vector.append(graph.features[feature])
        X.append(feature_vector)

    problem = PROBLEMS[problem_name].problem()

    for solver in solvers:
        places = []
        for i, graph in enumerate(dataset):
            place = 1
            for other_solver in solvers:
                if other_solver == solver:
                    continue
                cmp = problem.is_solution_worse(
                    solutions_logs[solver][i], solutions_logs[other_solver][i]
                )
                if cmp > 0:
                    place += 1
            places.append(place)

        print(f"Statistics for solver: {solver}")
        print(f"Average place: {sum(places) / len(places):.2f}")
        print(f"Best place count: {places.count(1)}")
        print(f"Worst place count: {places.count(len(solvers))}")

        # fit with at most 16 classes to avoid overcomplicating the tree
        clf = DecisionTreeClassifier(max_leaf_nodes=16)
        clf.fit(X, places)

        plt.figure(figsize=(20, 10))
        tree.plot_tree(
            clf,
            feature_names=features,
            class_names=[str(i) for i in range(1, len(solvers) + 1) if i in places],
            filled=True,
        )
        plt.savefig(
            os.path.join(
                ANALYSIS_FOLDER,
                f"{dataset_name}_{solver}_decision_tree.png",
            )
        )

    best_solvers = []
    for i, graph in enumerate(dataset):
        best_solver = None
        for solver in solvers:
            if best_solver is None:
                best_solver = solver
            else:
                cmp = problem.is_solution_worse(
                    solutions_logs[solver][i], solutions_logs[best_solver][i]
                )
                if cmp < 0:
                    best_solver = solver
        best_solvers.append(best_solver)

    clf = DecisionTreeClassifier()
    clf.fit(X, best_solvers)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clf,
        feature_names=features,
        class_names=[solver for solver in solvers if solver in best_solvers],
        filled=True,
    )
    plt.savefig(
        os.path.join(
            ANALYSIS_FOLDER,
            f"{dataset_name}_best_solver_decision_tree.png",
        )
    )


if __name__ == "__main__":
    if not os.path.exists(ANALYSIS_FOLDER):
        os.makedirs(ANALYSIS_FOLDER)

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        datasets = list_datasets()
        dataset = questionary.select(
            "Select a dataset to analyze", choices=datasets
        ).ask()

    loaded_dataset = load_dataset(dataset)

    if len(sys.argv) > 2:
        problem_name = sys.argv[2]
    else:
        problems = list_problems(dataset)
        problem_name = questionary.select(
            "Select a problem to analyze", choices=problems
        ).ask()

    if len(sys.argv) > 3:
        chosen_solvers = sys.argv[3].split(",")
    else:
        solutions = list_solutions(dataset, problem_name)
        if len(solutions) == 0:
            print("Can't analyze dataset with no solutions")
            exit()
        chosen_solvers = []
        while len(chosen_solvers) == 0:
            chosen_solvers = questionary.checkbox(
                "Choose solvers to analyze",
                choices=solutions,
            ).ask()
            if len(chosen_solvers) == 0:
                print("Need to choose solutions to analyze")

    if len(sys.argv) > 4:
        chosen_features = sys.argv[4].split(",")
    else:
        fcf = fully_calculated_features(loaded_dataset)
        if len(fcf) == 0:
            print("Can't analyze without calculated features")
        chosen_features = []
        while chosen_features == []:
            chosen_features = questionary.checkbox(
                "Choose features to analyze", fcf
            ).ask()
            if len(chosen_features) == 0:
                print("Need to choose features to analyze")

    analyzer(dataset, loaded_dataset, problem_name, chosen_solvers, chosen_features)
