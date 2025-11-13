import os
import sys
from pathlib import Path

import questionary
import yaml
from tqdm import tqdm

import framework.dataset_creators  # noqa: F401
import framework.feature_extractors  # noqa: F401
import framework.solvers  # noqa: F401
from framework.core.graph import Dataset
from framework.core.registries import DATASET_CREATORS, FEATURE_EXTRACTORS, SOLVERS
from framework.experiment.analyzer import analyzer
from framework.experiment.config import SOLUTIONS_FOLDER
from framework.experiment.utils import (
    fully_calculated_features,
    load_dataset,
    save_dataset,
    save_dataset_with_name,
    save_solver_solution,
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


def validate_config(config: dict) -> None:
    required_keys = ["datasets", "solvers", "feature_extractors"]
    for key in required_keys:
        if key not in config or not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"Missing or invalid required key: {key}")


def solution_exists(solver_name: str, dataset_name: str) -> bool:
    solution_path = os.path.join(SOLUTIONS_FOLDER, dataset_name, f"{solver_name}.csv")
    return os.path.exists(solution_path)


def main(config_path: Path):
    config = load_config(config_path)
    validate_config(config)

    if len(config["datasets"]) == 1 and config["datasets"][0]["type"] == "loaded":
        dataset_name = config["datasets"][0]["name"]
        if dataset_name.endswith(".pkl"):
            dataset_name = dataset_name[:-4]
        dataset = load_dataset(dataset_name)
        print(f"Loaded dataset {dataset_name} with {len(dataset)} graphs.")
    else:
        dataset = Dataset([])
        print("Loading datasets...")
        for dataset_cfg in tqdm(config["datasets"]):
            if dataset_cfg["type"] == "loaded":
                if dataset_cfg["name"].endswith(".pkl"):
                    dataset_cfg["name"] = dataset_cfg["name"][:-4]
                loaded_dataset = load_dataset(dataset_cfg["name"])
                dataset.extend(loaded_dataset)
            else:
                creator_fn = DATASET_CREATORS.get(dataset_cfg["generator"])
                creator_instance = creator_fn()
                generated_dataset = creator_instance.create_dataset(
                    dataset_cfg.get("parameters", {})
                )
                dataset.extend(generated_dataset)

        dataset_name = save_dataset(dataset)
        print(f"Dataset saved as {dataset_name} with {len(dataset)} graphs.")

    if "feature_extractors" in config:
        print("Calculating features...")
        fcf = fully_calculated_features(dataset)
        for feature_extractor_cfg in config["feature_extractors"]:
            print(f"Calculating features with {feature_extractor_cfg['name']}...")
            extractor_fn = FEATURE_EXTRACTORS.get(feature_extractor_cfg["name"])
            extractor_instance = extractor_fn()

            if all(
                feature in fcf for feature in extractor_instance.feature_names()
            ) and not feature_extractor_cfg.get("overwrite", False):
                print(
                    f"All features from extractor {feature_extractor_cfg['name']} already calculated for all graphs. Skipping..."
                )
                continue

            for graph in tqdm(dataset):
                calculated_features = extractor_instance.extract_features(graph)
                for feature in calculated_features:
                    graph.add_feature(
                        feature, overwrite=feature_extractor_cfg.get("overwrite", False)
                    )
            save_dataset_with_name(dataset, dataset_name)

        print("Feature extraction completed.")

    print("Using solvers...")
    for solver_cfg in config["solvers"]:
        if not solver_cfg.get("overwrite", False) and solution_exists(
            solver_cfg["name"], dataset_name
        ):
            print(
                f"Solutions for solver {solver_cfg['name']} on dataset {dataset_name} already exist. Skipping..."
            )
            continue
        print(f"Solving with {solver_cfg['name']}...")
        solver_fn = SOLVERS.get(solver_cfg["name"])
        solver_instance = solver_fn()
        solutions = []
        for graph in tqdm(dataset):
            solutions.append(solver_instance.solve(graph))
        solution_file = save_solver_solution(
            solver_cfg["name"], solutions, dataset_name
        )
        print(f"Solutions for solver {solver_cfg['name']} saved to {solution_file}.")

    if "analyze" in config:
        print("Analyzing results...")
        chosen_solvers = [solver_cfg["name"] for solver_cfg in config["solvers"]]
        features = config["analyze"].get("features", [])
        analyzer(dataset_name, dataset, chosen_solvers, features)

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
    main(os.path.join(CONFIGS_FOLDER, config))
