import yaml
import os
from pathlib import Path
import sys
import questionary

import framework.dataset_creators  # noqa: F401
import framework.feature_extractors  # noqa: F401
import framework.solvers  # noqa: F401

from framework.experiment.utils import load_dataset, save_dataset, save_dataset_with_name, save_solver_solution
from framework.core.graph import Dataset
from framework.core.registries import DATASET_CREATORS, FEATURE_EXTRACTORS, SOLVERS

CONFIGS_FOLDER = 'config/'

def list_configs() -> list[str]:
    configs = []
    for file in os.listdir(CONFIGS_FOLDER):
        if file.endswith(".yaml"):
            configs.append(file[:-5])
    return configs

def load_config(config_path: Path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config: dict) -> None:
    required_keys = ['datasets', 'solvers', 'feature_extractors']
    for key in required_keys:
        if key not in config or not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"Missing or invalid required key: {key}")
        
def main(config_path: Path):
    config = load_config(config_path)
    validate_config(config)

    dataset = Dataset([])
    for dataset_cfg in config['datasets']:
        if dataset_cfg["type"] == "loaded":
            if dataset_cfg["name"].endswith(".pkl"):
                dataset_cfg["name"] = dataset_cfg["name"][:-4]
            loaded_dataset = load_dataset(dataset_cfg["name"])
            dataset.extend(loaded_dataset)
        else:
            creator_fn = DATASET_CREATORS.get(dataset_cfg["generator"])
            creator_instance = creator_fn()
            generated_dataset = creator_instance.create_dataset(dataset_cfg.get("parameters", {}))
            dataset.extend(generated_dataset)
    
    dataset_name = save_dataset(dataset)
    print(f"Dataset saved as {dataset_name} with {len(dataset)} graphs.")

    for feature_extractor_cfg in config['feature_extractors']:
        extractor_fn = FEATURE_EXTRACTORS.get(feature_extractor_cfg["name"])
        extractor_instance = extractor_fn()
        for graph in dataset:
            calculated_features = extractor_instance.extract_features(graph)
            for feature in calculated_features:
                graph.add_feature(feature, overwrite=feature_extractor_cfg.get("overwrite", False))
        save_dataset_with_name(dataset, dataset_name)
     
    print("Feature extraction completed.")

    for solver_cfg in config['solvers']:
        solver_fn = SOLVERS.get(solver_cfg["name"])
        solver_instance = solver_fn()
        solutions = [solver_instance.solve(graph) for graph in dataset]
        solution_file = save_solver_solution(solver_cfg["name"], solutions, dataset_name)
        print(f"Solutions for solver {solver_cfg['name']} saved to {solution_file}.")

    print("All tasks completed.")

if __name__ == "__main__":
    all_configs = list_configs()
    if len(sys.argv) != 2:
        config = questionary.select(
            "Select a config to run:", choices=all_configs
        ).ask() + ".yaml"
    else:
        config = sys.argv[1]
        if not config.endswith(".yaml"):
            config += ".yaml"
    main(os.path.join(CONFIGS_FOLDER, config))
            
