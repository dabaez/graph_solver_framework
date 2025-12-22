import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import questionary
from scipy.stats import pearsonr

from framework.core.graph import Dataset
from framework.core.registries import (
    FEATURE_EXTRACTORS,
    GRAPH_CREATORS,
    SOLVERS,
)
from framework.dataset.SQLiteDataset import SQLiteDataset

from ..core.solver import Solution
from .config import DATASETS_FOLDER, SOLUTIONS_FOLDER

##### REGISTRIES #####


def list_registered_graph_creators():
    """List all registered graph creators."""
    return list(GRAPH_CREATORS.keys())


def list_registered_dataset_solvers():
    """List all registered dataset solvers."""
    return list(SOLVERS.keys())


def list_registered_dataset_feature_extractors():
    """List all registered dataset feature extractors."""
    return list(FEATURE_EXTRACTORS.keys())


##### DATASET #####


def list_datasets() -> list[str]:
    datatsets = []
    for file in os.listdir(DATASETS_FOLDER):
        if file.endswith(".db"):
            datatsets.append(file[:-3])
    return datatsets


def load_dataset(dataset_name: str) -> Dataset:
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.db")
    return SQLiteDataset.from_file(file_path=file_path)


def CalculatedFeaturesFromDataset(dataset: Dataset) -> dict[str, int]:
    """Calculate for each feature in the dataset the number of graphs that have that feature."""
    feature_counts = {}
    for graph in dataset:
        for feature in graph.features:
            if feature not in feature_counts:
                feature_counts[feature] = 0
            feature_counts[feature] += 1
    return feature_counts


def CalculatedFeaturesPercentageFromDataset(dataset: Dataset) -> dict[str, float]:
    """Calculate for each feature in the dataset the percentage of graphs that have that feature."""
    total_graphs = len(dataset)
    feature_counts = CalculatedFeaturesFromDataset(dataset)
    return {name: count / total_graphs * 100 for name, count in feature_counts.items()}


def fully_calculated_features(dataset: Dataset) -> list[str]:
    return [
        name
        for name, count in CalculatedFeaturesFromDataset(dataset).items()
        if count == len(dataset)
    ]


def dataset_exists(dataset_name: str) -> bool:
    """Check if a dataset exists in the datasets folder."""
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.db")
    return os.path.exists(file_path)


def extend_dataset(dataset: Dataset, additional_datasets: list[str]) -> None:
    """Extend a dataset with additional datasets."""
    with dataset.writer() as writer:
        for name in additional_datasets:
            additional_dataset = load_dataset(name)
            for graph in additional_dataset:
                writer.add(graph)


def merge_datasets(dataset_names: list[str], new_dataset_name: str) -> Dataset:
    """Merge multiple datasets into a new dataset."""
    merged_dataset = SQLiteDataset.from_file(
        file_path=os.path.join(DATASETS_FOLDER, f"{new_dataset_name}.db")
    )
    extend_dataset(merged_dataset, dataset_names)
    return merged_dataset


def delete_dataset(dataset_name: str) -> None:
    """Delete a dataset from the datasets folder."""
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.db")
    os.remove(file_path)


def rename_dataset(old_name: str, new_name: str) -> None:
    """Rename a dataset in the datasets folder."""
    old_path = os.path.join(DATASETS_FOLDER, f"{old_name}.db")
    new_path = os.path.join(DATASETS_FOLDER, f"{new_name}.db")
    os.rename(old_path, new_path)


def calculate_feature_correlation_matrix(
    dataset: Dataset, features: list[str]
) -> list[list[float]]:
    """
    Calculate the Pearson correlation matrix for the features in the dataset.

    Args:
        dataset (Dataset): The dataset containing features.
        features (list[str]): List of feature names to calculate correlation for.

    Returns:
        list[list[float]]: A 2D list representing the correlation matrix.
    """
    feature_arrays = [[] for _ in features]

    for graph in dataset:
        for i, feature_name in enumerate(features):
            feature_arrays[i].append(graph.features[feature_name])

    correlation_matrix = [[0.0] * len(features) for _ in range(len(features))]

    for i in range(len(features)):
        correlation_matrix[i][i] = 1.0
        for j in range(i + 1, len(features)):
            corr, _ = pearsonr(feature_arrays[i], feature_arrays[j])
            correlation_matrix[i][j] = corr  # type: ignore
            correlation_matrix[j][i] = corr  # type: ignore

    return correlation_matrix


def plot_feature_correlation_matrix(
    correlation_matrix: list[list[float]], features: list[str], pathname: str
) -> None:
    """Plot the feature correlation matrix using matplotlib."""

    plt.figure(figsize=(max(8, len(features) * 0.5), max(6, len(features) * 0.5)))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation Coefficient")
    plt.xticks(ticks=np.arange(len(features)), labels=features, rotation=90)
    plt.yticks(ticks=np.arange(len(features)), labels=features)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(pathname)
    plt.close()


def get_valid_dataset_name_from_user() -> str | None:
    """Prompt the user to enter a valid dataset name."""

    while True:
        new_name = questionary.text(
            "Enter a name for the dataset (or leave empty to go back):"
        ).ask()
        if not new_name:
            return None
        if dataset_exists(new_name):
            print(
                f"Dataset '{new_name}' already exists. Please choose a different name."
            )
        else:
            return new_name


##### SOLVERS #####


def save_solver_solution(
    solver_name: str, solution: list[Solution], dataset_name: str
) -> str:
    """Save a solver's solution to the solutions folder."""
    dir_path = os.path.join(SOLUTIONS_FOLDER, dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{solver_name}.csv")
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Graph Index", "Solution length", "Time taken", "Solution"])
        for i, sol in enumerate(solution):
            writer.writerow(
                [
                    i + 1,
                    len(sol.mis),
                    sol.time,
                    ", ".join(sol.mis),
                ]
            )
    return file_path


##### MISC #####


def create_folder_if_not_exists(folder_path: str):
    """Create a folder if it does not exist."""
    import os

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
