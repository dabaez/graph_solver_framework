import os
import sys
from pathlib import Path

import questionary
import yaml
from tqdm import tqdm

import framework.feature_extractors  # noqa: F401
import framework.graph_creators  # noqa: F401
import framework.solvers  # noqa: F401
from framework.core.registries import FEATURE_EXTRACTORS, GRAPH_CREATORS, SOLVERS
from framework.dataset.SQLiteDataset import SQLiteDataset
from framework.experiment.analyzer import analyzer
from framework.experiment.config import SOLUTIONS_FOLDER
from framework.experiment.utils import (
    extend_dataset_with_path,
    fully_calculated_features,
    load_dataset,
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
    config_filename = os.path.basename(config_path)[:-5]

    validate_config(config)

    if len(config["datasets"]) == 1 and config["datasets"][0]["type"] == "loaded":
        dataset_name = config["datasets"][0]["name"]
        if dataset_name.endswith(".db"):
            dataset_name = dataset_name[:-3]
        dataset = load_dataset(dataset_name)
        print(f"Loaded dataset {dataset_name} with {len(dataset)} graphs.")
    else:
        dataset_name = config_filename + ".db"
        dataset = SQLiteDataset.from_file(dataset_name)
        if len(dataset) > 0:
            print(
                f"Dataset {dataset_name} already exists with {len(dataset)} graphs. Skipping creation."
            )
        else:
            print("Loading datasets...")
            for dataset_cfg in tqdm(config["datasets"]):
                if dataset_cfg["type"] == "loaded":
                    if dataset_cfg["name"].endswith(".db"):
                        dataset_cfg["name"] = dataset_cfg["name"][:-3]
                    extend_dataset_with_path(dataset, [dataset_cfg["name"]])
                else:
                    creator_fn = GRAPH_CREATORS[dataset_cfg["generator"]]
                    creator_instance = creator_fn()
                    creator_instance.create_graphs(
                        dataset_cfg.get("parameters", {}), dataset
                    )

            print(f"Dataset saved as {dataset_name} with {len(dataset)} graphs.")

    if "feature_extractors" in config:
        print("Calculating features...")
        fcf = fully_calculated_features(dataset)
        for feature_extractor_cfg in config["feature_extractors"]:
            print(f"Calculating features with {feature_extractor_cfg['name']}...")
            extractor_fn = FEATURE_EXTRACTORS[feature_extractor_cfg["name"]]
            extractor_instance = extractor_fn()

            if all(
                feature in fcf for feature in extractor_instance.feature_names()
            ) and not feature_extractor_cfg.get("overwrite", False):
                print(
                    f"All features from extractor {feature_extractor_cfg['name']} already calculated for all graphs. Skipping..."
                )
                continue

            with dataset.writer() as writer:
                for graph in tqdm(dataset):
                    with graph as G:
                        calculated_features = extractor_instance.extract_features(G)
                    updated_features = False
                    for feature in calculated_features:
                        updated = graph.add_feature(
                            feature.name,
                            feature.value,
                            overwrite=feature_extractor_cfg.get("overwrite", False),
                        )
                        updated_features = updated_features or updated
                    if updated_features:
                        writer.update_features(graph)

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
        solver_fn = SOLVERS[solver_cfg["name"]]
        solver_instance = solver_fn()
        solutions = []
        for graph in tqdm(dataset):
            with graph as G:
                solutions.append(solver_instance.solve(G))
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
    main(Path(os.path.join(CONFIGS_FOLDER, config)))
