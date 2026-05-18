import os
import sys
from pathlib import Path

import questionary
import yaml
from tqdm import tqdm

import feature_extractors  # noqa: F401
import graph_creators  # noqa: F401
import problems  # noqa: F401
from framework.core.registries import FEATURE_EXTRACTORS, GRAPH_CREATORS, SOLVERS
from framework.dataset.SQLiteDataset import SQLiteDataset
from framework.experiment.analyzer import analyzer
from framework.experiment.config import DATASETS_FOLDER
from framework.experiment.utils import (
    extend_dataset_with_path,
    list_datasets,
    load_dataset,
    save_solver_solution,
    solution_exists,
)

CONFIGS_FOLDER = "config/"


def list_configs() -> list[str]:
    configs = []
    for file in os.listdir(CONFIGS_FOLDER):
        if file.endswith(".yaml"):
            configs.append(file[:-5])
    return configs


def load_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def validate_datasets(config: dict) -> None:
    available_datasets = list_datasets()
    available_creators = GRAPH_CREATORS.keys()
    for dataset_cfg in config["datasets"]:
        if dataset_cfg["type"] == "loaded":
            dataset_name = dataset_cfg["name"]
            if dataset_name.endswith(".db"):
                dataset_name = dataset_name[:-3]
            if dataset_name not in available_datasets:
                raise ValueError(f"Dataset {dataset_name} not found.")
        else:
            generator_name = dataset_cfg["generator"]
            if generator_name not in available_creators:
                raise ValueError(f"Graph creator {generator_name} not found.")
            creator_instance = GRAPH_CREATORS[generator_name]()
            if not creator_instance.validate_parameters(
                dataset_cfg.get("parameters", {})
            ):
                raise ValueError(
                    f"Invalid parameters for graph creator {generator_name}."
                )


def validate_config(config: dict) -> None:
    if (
        "datasets" not in config
        or not isinstance(config["datasets"], list)
        or not config["datasets"]
    ):
        raise ValueError("Missing or invalid required key: datasets")

    if "problem_name" not in config and "solvers" in config:
        raise ValueError("Solvers defined but no problem_name specified in config.")

    if "analyze" in config and "solvers" not in config:
        raise ValueError("Analyze section found but no solvers defined in config.")

    validate_datasets(config)


def load_or_create_dataset(config: dict, config_filename: str):
    if len(config["datasets"]) == 1 and config["datasets"][0]["type"] == "loaded":
        dataset_name = config["datasets"][0]["name"]
        if dataset_name.endswith(".db"):
            dataset_name = dataset_name[:-3]
        dataset = load_dataset(dataset_name)
        print(f"Loaded dataset {dataset_name} with {len(dataset)} graphs.")
        return dataset, dataset_name

    dataset_name = config_filename
    dataset_path = os.path.join(DATASETS_FOLDER, f"{dataset_name}.db")
    dataset = SQLiteDataset.from_file(dataset_path)

    if len(dataset) > 0:
        print(
            f"Dataset {dataset_name} already exists with {len(dataset)} graphs. Skipping creation."
        )
        return dataset, dataset_name

    print("Loading datasets...")
    for dataset_cfg in tqdm(config["datasets"]):
        if dataset_cfg["type"] == "loaded":
            source_name = dataset_cfg["name"]
            if source_name.endswith(".db"):
                source_name = source_name[:-3]
            extend_dataset_with_path(dataset_name, dataset, [source_name])
        else:
            creator_fn = GRAPH_CREATORS[dataset_cfg["generator"]]
            creator_instance = creator_fn()
            dataset = creator_instance.create_graphs(
                dataset_cfg.get("parameters", {}), dataset
            )

    print(f"Dataset saved as {dataset_name} with {len(dataset)} graphs.")
    return dataset, dataset_name


def extract_features(config: dict, dataset) -> None:
    print("Calculating features...")
    for feature_extractor_cfg in config["feature_extractors"]:
        print(f"Calculating features with {feature_extractor_cfg['name']}...")
        extractor_fn = FEATURE_EXTRACTORS[feature_extractor_cfg["name"]]
        extractor_instance = extractor_fn()
        feature_names = extractor_instance.feature_names()
        overwrite_features = feature_extractor_cfg.get("overwrite", False)

        with dataset.writer() as writer:
            for graph in tqdm(dataset):
                missing_features = [
                    feature
                    for feature in feature_names
                    if feature not in graph.features
                ]
                if not missing_features and not overwrite_features:
                    continue
                with graph as G:
                    calculated_features = extractor_instance.extract_features(G)
                updated_features = False
                for feature in calculated_features:
                    updated = graph.add_feature(
                        feature.name,
                        feature.value,
                        overwrite=overwrite_features,
                    )
                    updated_features = updated_features or updated
                if updated_features:
                    writer.update_features(graph)

    print("Feature extraction completed.")


def run_solvers(config: dict, dataset, dataset_name: str) -> None:
    problem_name = config["problem_name"]
    print("Using solvers...")
    for solver_cfg in config["solvers"]:
        if not solver_cfg.get("overwrite", False) and solution_exists(
            problem_name, solver_cfg["name"], dataset_name
        ):
            print(
                f"Solutions for solver {solver_cfg['name']} on dataset {dataset_name} already exist. Skipping..."
            )
            continue
        print(f"Solving with {solver_cfg['name']}...")
        solver_fn = SOLVERS[problem_name][solver_cfg["name"]]
        solver_instance = solver_fn()
        solutions = {}
        for graph in tqdm(dataset):
            with graph as G:
                solutions[graph.metadata["uuid"]] = solver_instance.solve(G)
        solution_file = save_solver_solution(
            solver_cfg["name"], solutions, dataset_name, problem_name
        )
        print(f"Solutions for solver {solver_cfg['name']} saved to {solution_file}.")


def run_analysis(config: dict, dataset, dataset_name: str) -> None:
    print("Analyzing results...")
    chosen_solvers = [solver_cfg["name"] for solver_cfg in config["solvers"]]
    features = config["analyze"].get("features", [])
    analyzer(dataset_name, dataset, config["problem_name"], chosen_solvers, features)


def main(config_path: Path):
    config = load_config(config_path)
    config_filename = os.path.basename(config_path)[:-5]

    try:
        validate_config(config)
    except ValueError as e:
        print(f"Config validation error: {e}")
        return

    dataset, dataset_name = load_or_create_dataset(config, config_filename)

    if "feature_extractors" in config:
        extract_features(config, dataset)

    if "solvers" in config:
        run_solvers(config, dataset, dataset_name)

    if "analyze" in config and "solvers" in config:
        run_analysis(config, dataset, dataset_name)

    print("All tasks completed.")


if __name__ == "__main__":
    all_configs = list_configs()
    if len(sys.argv) != 2:
        config = (
            questionary.select("Select a config to run:", choices=all_configs).ask()
            + ".yaml"
        )
    else:
        config = sys.argv[1]
        if not config.endswith(".yaml"):
            config += ".yaml"
    main(Path(os.path.join(CONFIGS_FOLDER, config)))
