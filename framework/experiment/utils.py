from framework.core.graph import Dataset
from framework.core.registries import (
    DATASET_CREATORS,
    DATASET_FEATURE_EXTRACTORS,
    DATASET_SOLVERS,
)

##### REGISTRIES #####


def list_registered_dataset_creators():
    """List all registered dataset creators."""
    return list(DATASET_CREATORS.keys())


def list_registered_dataset_solvers():
    """List all registered dataset solvers."""
    return list(DATASET_SOLVERS.keys())


def list_registered_dataset_feature_extractors():
    """List all registered dataset feature extractors."""
    return list(DATASET_FEATURE_EXTRACTORS.keys())


##### DATASET #####


def list_datasets():
    

def CalculatedFeaturesFromDataset(dataset: Dataset) -> dict[str, int]:
    """Calculate for each feature in the dataset the number of graphs that have that feature."""
    feature_counts = {}
    for graph in dataset:
        for feature in graph.features:
            if feature.name not in feature_counts:
                feature_counts[feature.name] = 0
            feature_counts[feature.name] += 1
    return feature_counts


def CalculatedFeaturesPercentageFromDataset(dataset: Dataset) -> dict[str, float]:
    """Calculate for each feature in the dataset the percentage of graphs that have that feature."""
    total_graphs = len(dataset)
    feature_counts = CalculatedFeaturesFromDataset(dataset)
    return {name: count / total_graphs * 100 for name, count in feature_counts.items()}
