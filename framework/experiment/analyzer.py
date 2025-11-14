import csv
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import questionary
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from framework.core.graph import Dataset
from framework.experiment.config import ANALYSIS_FOLDER, SOLUTIONS_FOLDER
from framework.experiment.utils import (
    fully_calculated_features,
    list_datasets,
    load_dataset,
)


def list_solutions(dataset: str):
    solutions_path = os.path.join(SOLUTIONS_FOLDER, dataset)
    solutions = []
    for file in os.listdir(solutions_path):
        if file.endswith(".csv"):
            solutions.append(file[:-4])
    return solutions


@dataclass
class SolutionLog:
    graph_index: int
    solution_length: int
    time_taken: float
    solution: list[str]


def load_solution(dataset_name: str, solver: str) -> list[SolutionLog]:
    with open(os.path.join(SOLUTIONS_FOLDER, dataset_name, f"{solver}.csv")) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        return [
            SolutionLog(
                graph_index=int(row[0]),
                solution_length=int(row[1]),
                time_taken=float(row[2]),
                solution=map(str.strip, row[3].split(",")),
            )
            for row in reader
        ]


def compare_solutions(sol1: SolutionLog, sol2: SolutionLog) -> int:
    if sol1.solution_length > sol2.solution_length:
        return -1
    elif sol1.solution_length < sol2.solution_length:
        return 1
    else:
        return sol1.time_taken - sol2.time_taken


def analyzer(
    dataset_name: str, dataset: Dataset, solvers: list[str], features: list[str]
):
    solutions_logs = {}
    for solver in solvers:
        solutions_logs[solver] = load_solution(dataset_name, solver)
        assert len(solutions_logs[solver]) == len(dataset), (
            f"Solution log for {solver} is not complete"
        )

    X = []
    for i, graph in enumerate(dataset):
        feature_vector = []
        for feature in features:
            feature_vector.append(graph.features[feature])
        X.append(feature_vector)

    for solver in solvers:
        places = []
        for i, graph in enumerate(dataset):
            place = 1
            for other_solver in solvers:
                if other_solver == solver:
                    continue
                cmp = compare_solutions(
                    solutions_logs[solver][i], solutions_logs[other_solver][i]
                )
                if cmp > 0:
                    place += 1
            places.append(place)

        print(f"Statistics for solver: {solver}")
        print(f"Average place: {sum(places) / len(places):.2f}")
        print(f"Best place count: {places.count(1)}")
        print(f"Worst place count: {places.count(len(solvers))}")

        clf = DecisionTreeClassifier()
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
                cmp = compare_solutions(
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
        chosen_solvers = sys.argv[2].split(",")
    else:
        solutions = list_solutions(dataset)
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

    if len(sys.argv) > 3:
        chosen_features = sys.argv[3].split(",")
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

    analyzer(dataset, loaded_dataset, chosen_solvers, chosen_features)
