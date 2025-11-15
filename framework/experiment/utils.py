import csv
import os
import pickle

from framework.core.graph import Dataset
from framework.core.registries import (
    DATASET_CREATORS,
    FEATURE_EXTRACTORS,
    SOLVERS,
)

from ..core.solver import Solution
from .config import DATASETS_FOLDER, SOLUTIONS_FOLDER

##### REGISTRIES #####


def list_registered_dataset_creators():
    """List all registered dataset creators."""
    return list(DATASET_CREATORS.keys())


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
        if file.endswith(".pkl"):
            datatsets.append(file[:-4])
    return datatsets


def load_dataset(dataset_name: str) -> Dataset:
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.pkl")
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_dataset(dataset: Dataset) -> str:
    """Save a dataset to the datasets folder."""
    file_name = f"dataset_{len(list_datasets())}.pkl"
    file_path = os.path.join(DATASETS_FOLDER, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)
    return file_name[:-4]


def save_dataset_with_name(dataset: Dataset, dataset_name: str) -> str:
    """Save a dataset to the datasets folder with a specific name."""
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)
    return dataset_name


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
    return [name for name,count in CalculatedFeaturesFromDataset(dataset).items() if count == len(dataset)]

def dataset_exists(dataset_name: str) -> bool:
    """Check if a dataset exists in the datasets folder."""
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.pkl")
    return os.path.exists(file_path)


def merge_datasets(dataset_names: list[str], new_dataset_name: str) -> Dataset:
    """Merge multiple datasets into a new dataset."""
    merged_dataset = Dataset([])
    for name in dataset_names:
        dataset = load_dataset(name)
        merged_dataset.extend(dataset)

    save_dataset_with_name(merged_dataset, new_dataset_name)
    return merged_dataset


def delete_dataset(dataset_name: str) -> None:
    """Delete a dataset from the datasets folder."""
    file_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.pkl")
    os.remove(file_path)


def rename_dataset(old_name: str, new_name: str) -> None:
    """Rename a dataset in the datasets folder."""
    old_path = os.path.join(DATASETS_FOLDER, f"{old_name}.pkl")
    new_path = os.path.join(DATASETS_FOLDER, f"{new_name}.pkl")
    os.rename(old_path, new_path)


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
