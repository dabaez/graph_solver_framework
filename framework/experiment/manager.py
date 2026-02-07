import os

import numpy as np
import questionary
from tqdm import tqdm

import _bootstrap  # noqa: F401
from framework.core.registries import (
    FEATURE_EXTRACTORS,
    GRAPH_CREATORS,
    SOLVERS,
)

from .config import DATA_FOLDER, DATASETS_FOLDER, SOLUTIONS_FOLDER
from .utils import (
    CalculatedFeaturesFromDataset,
    CalculatedFeaturesPercentageFromDataset,
    calculate_feature_correlation_matrix,
    create_folder_if_not_exists,
    dataset_exists,
    delete_dataset,
    extend_dataset_with_path,
    get_valid_dataset_name_from_user,
    list_datasets,
    list_registered_dataset_feature_extractors,
    list_registered_dataset_solvers,
    list_registered_graph_creators,
    list_registered_problems,
    load_dataset,
    merge_datasets,
    plot_feature_correlation_matrix,
    rename_dataset,
    save_solver_solution,
)


def initialize_folders():
    create_folder_if_not_exists(DATA_FOLDER)
    create_folder_if_not_exists(DATASETS_FOLDER)
    create_folder_if_not_exists(SOLUTIONS_FOLDER)


def explore_datasets():
    available_datasets = list_datasets()
    if not available_datasets:
        print("No datasets available.")
        return
    chosen_option = questionary.select(
        "Choose an option:",
        choices=[
            "Explore a specific dataset",
            "Merge datasets",
            "Delete datasets",
            "Go back",
        ],
    ).ask()
    if chosen_option == "Explore a specific dataset":
        chosen_dataset = questionary.select(
            "Choose a dataset to explore:", choices=[*available_datasets, "Go back"]
        ).ask()
        if chosen_dataset == "Go back":
            return
        explore_dataset(chosen_dataset)
    elif chosen_option == "Merge datasets":
        datatsets_to_merge = questionary.checkbox(
            "Select datasets to merge:", choices=available_datasets + ["Go back"]
        ).ask()
        if "Go back" in datatsets_to_merge:
            return
        if len(datatsets_to_merge) < 2:
            print("Can't merge less than two datasets.")
            return
        while True:
            new_dataset_name = questionary.text(
                "Enter a name for the new merged dataset (or leave empty to go back):"
            ).ask()
            if not new_dataset_name:
                return
            if dataset_exists(new_dataset_name):
                print(
                    f"Dataset '{new_dataset_name}' already exists. Please choose a different name."
                )
            else:
                break
        merge_datasets(datatsets_to_merge, new_dataset_name)
        print(f"Merged datasets into '{new_dataset_name}'.")
    elif chosen_option == "Delete datasets":
        datasets_to_delete = questionary.checkbox(
            "Select datasets to delete:", choices=available_datasets + ["Go back"]
        ).ask()
        if "Go back" in datasets_to_delete:
            return
        if not datasets_to_delete:
            print("No datasets selected for deletion.")
            return
        if questionary.confirm(
            f"You are about to delete {len(datasets_to_delete)} datasets. Are you sure?"
        ).ask():
            for dataset in datasets_to_delete:
                delete_dataset(dataset)
            print(f"Deleted datasets: {', '.join(datasets_to_delete)}")


def explore_dataset(dataset_name: str):
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return
    calculated_features = CalculatedFeaturesPercentageFromDataset(dataset)
    print(f"Dataset: {dataset_name}")
    print(f"Number of graphs: {len(dataset)}")
    if calculated_features:
        fully_calculated_features = [
            feature
            for feature, percentage in calculated_features.items()
            if percentage == 100.0
        ]
        non_fully_calculated_features = [
            f"{feature} ({percentage:.2f}%)"
            for feature, percentage in calculated_features.items()
            if percentage < 100.0
        ]
        if fully_calculated_features:
            print(
                f"Features fully calculated for all graphs: {', '.join(fully_calculated_features)}"
            )
        if non_fully_calculated_features:
            print(
                f"Features partially calculated: {', '.join(non_fully_calculated_features)}"
            )
    else:
        print("No calculated features found in this dataset.")
    chosen_option = questionary.select(
        "Choose an option:",
        choices=[
            "Explore graphs in this dataset",
            "Rename this dataset",
            "Extend with other datasets",
            "Delete this dataset",
            "Get feature correlation matrix",
            "Go back",
        ],
    ).ask()
    if chosen_option == "Explore graphs in this dataset":
        while True:
            graph_index = questionary.text(
                f"Choose a graph index from 1 to {len(dataset)} (or leave empty to go back):"
            ).ask()
            if not graph_index:
                break
            try:
                graph_index = int(graph_index) - 1
                graph = dataset[graph_index]
                print(f"Graph {graph_index + 1}:")
                if graph.features:
                    for feature, feature_value in graph.features.items():
                        print(f"Feature: {feature}, Value: {feature_value}")
                else:
                    print("No features found for this graph.")
                graph_representation = questionary.confirm(
                    "Do you want to see the graph representation?", default=False
                ).ask()
                if graph_representation:
                    with graph as g:
                        print(g.edges())
            except (ValueError, IndexError):
                print("Invalid graph index. Please try again.")
    elif chosen_option == "Rename this dataset":
        new_name = get_valid_dataset_name_from_user()
        if new_name:
            rename_dataset(dataset_name, new_name)
            print(f"Dataset renamed to '{new_name}'.")
            explore_dataset(new_name)
    elif chosen_option == "Extend with other datasets":
        available_datasets = [d for d in list_datasets() if d != dataset_name]
        datasets_to_extend = questionary.checkbox(
            "Select datasets to extend with:", choices=available_datasets + ["Go back"]
        ).ask()
        if "Go back" in datasets_to_extend:
            return
        if not datasets_to_extend:
            print("No datasets selected for extension.")
            return
        extend_dataset_with_path(dataset, datasets_to_extend)
    elif chosen_option == "Delete this dataset":
        if questionary.confirm(
            f"Are you sure you want to delete the dataset '{dataset_name}'?"
        ).ask():
            delete_dataset(dataset_name)
            print(f"Dataset '{dataset_name}' deleted.")
    elif chosen_option == "Get feature correlation matrix":
        chosen_features = questionary.checkbox(
            "Select features to include in the correlation matrix (choose none to go back):",
            choices=list(fully_calculated_features),
        ).ask()
        if not chosen_features:
            return
        correlation_matrix = calculate_feature_correlation_matrix(
            dataset, chosen_features
        )

        top_correlated_pairs = []
        for i in range(len(chosen_features)):
            for j in range(i + 1, len(chosen_features)):
                corr_value = correlation_matrix[i][j]
                if not np.isnan(corr_value):
                    top_correlated_pairs.append(
                        (chosen_features[i], chosen_features[j], corr_value)
                    )
        top_correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print("Top correlated feature pairs:")
        for feature1, feature2, corr_value in top_correlated_pairs[
            : 2 * len(chosen_features)
        ]:
            print(f"{feature1} - {feature2}: {corr_value:.4f}")

        filename = f"{dataset_name}_feature_correlation_matrix.png"
        path = os.path.join(DATA_FOLDER, filename)
        plot_feature_correlation_matrix(correlation_matrix, chosen_features, path)
        print(f"Feature correlation matrix saved as '{filename}'.")


def explore_graph_creators():
    creators = list_registered_graph_creators()
    if not creators:
        print("No graph creators available.")
        return
    chosen_graph_creator = questionary.select(
        "Choose a graph creator to explore:", choices=[*creators, "Go back"]
    ).ask()
    if chosen_graph_creator == "Go back":
        return
    explore_graph_creator(chosen_graph_creator)


def explore_graph_creator(graph_creator_name: str):
    creator_class = GRAPH_CREATORS.get(graph_creator_name)
    if not creator_class:
        print(f"Graph creator '{graph_creator_name}' not found.")
        return
    print(f"Graph Creator: {graph_creator_name}")
    creator_instance = creator_class()
    print(f"Description: {creator_instance.description()}")
    required_params = creator_instance.required_parameters()
    if required_params:
        print("Required parameters:")
        for param in required_params:
            print(f"- {param.name}: {param.description}")
    else:
        print("This graph creator does not require any parameters.")

    chosen_option = questionary.select(
        "Choose an option:",
        choices=[
            "Create graphs on a new dataset",
            "Add graphs to an existing dataset",
            "Go back",
        ],
    ).ask()

    if chosen_option == "Create graphs on a new dataset":
        dataset_name = get_valid_dataset_name_from_user()
        if not dataset_name:
            return
        dataset = load_dataset(dataset_name)
    elif chosen_option == "Add graphs to an existing dataset":
        existing_datasets = list_datasets()
        if not existing_datasets:
            print("No existing datasets available to add graphs to.")
            return
        dataset_name = questionary.select(
            "Choose an existing dataset to add graphs to:",
            choices=existing_datasets + ["Go back"],
        ).ask()
        if dataset_name == "Go back":
            return
        dataset = load_dataset(dataset_name)

    if (
        chosen_option == "Create graphs on a new dataset"
        or chosen_option == "Add graphs to an existing dataset"
    ):
        parameters = {param.name: "" for param in required_params}
        while True:
            for param in required_params:
                if param.isPath:
                    value = questionary.path(
                        f"{param.name} (path):", default=parameters[param.name]
                    ).ask()
                else:
                    value = questionary.text(
                        f"{param.name}:", default=parameters[param.name]
                    ).ask()
                parameters[param.name] = value

            if creator_instance.validate_parameters(parameters):
                break

            else:
                print("Invalid parameters provided.")
                if not questionary.confirm("Do you want to try again?").ask():
                    return

        creator_instance.create_graphs(parameters=parameters, dataset=dataset)
        print(f"Graphs created and added to the dataset '{dataset_name}'.")


def explore_feature_extractors():
    feature_extractors = list_registered_dataset_feature_extractors()
    if not feature_extractors:
        print("No dataset feature extractors available.")
        return
    chosen_extractor = questionary.select(
        "Choose a dataset feature extractor to explore:",
        choices=[*feature_extractors, "Go back"],
    ).ask()
    if chosen_extractor == "Go back":
        return
    explore_feature_extractor(chosen_extractor)


def explore_feature_extractor(extractor_name: str):
    extractor_class = FEATURE_EXTRACTORS.get(extractor_name)
    if not extractor_class:
        print(f"Feature extractor '{extractor_name}' not found.")
        return
    print(f"Feature Extractor: {extractor_name}")
    extractor_instance = extractor_class()
    print(f"Description: {extractor_instance.description()}")
    feature_names = extractor_instance.feature_names()
    if feature_names:
        print("Features to be extracted: " + ", ".join(feature_names))
    dataset_name = questionary.select(
        "Choose a dataset to extract features from:",
        choices=list_datasets() + ["Go back"],
    ).ask()
    if dataset_name == "Go back":
        return
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return
    present_features = CalculatedFeaturesFromDataset(dataset)
    overwrite_features = True
    if any(feature in feature_names for feature in present_features):
        overwrite_features = questionary.confirm(
            "Some features are already present in the dataset. Do you want to overwrite them? 'Y' for overwrite, 'N' to skip those features."
        ).ask()
    with dataset.writer() as writer:
        for graph in tqdm(dataset):
            missing_features = [
                feature for feature in feature_names if feature not in graph.features
            ]
            if not missing_features and not overwrite_features:
                continue
            with graph as g:
                calculated_features = extractor_instance.extract_features(g)
                updated = False
            for feature in calculated_features:
                updated_feature = graph.add_feature(
                    feature.name, feature.value, overwrite=overwrite_features
                )
                updated = updated or updated_feature
            if updated:
                writer.update_features(graph)
    print(f"Features extracted and added to the dataset '{dataset_name}'.")


def explore_solvers():
    problems = list_registered_problems()
    if not problems:
        print("No problems available.")
        return
    chosen_problem = questionary.select(
        "Choose a problem to explore solvers for:",
        choices=[*problems, "Go back"],
    ).ask()
    if chosen_problem == "Go back":
        return
    solvers = list_registered_dataset_solvers(chosen_problem)
    if not solvers:
        print("No dataset solvers available.")
        return
    chosen_solver = questionary.select(
        "Choose a dataset solver to explore:",
        choices=[*solvers, "Go back"],
    ).ask()
    if chosen_solver == "Go back":
        return
    explore_solver(chosen_problem, chosen_solver)


def explore_solver(problem_name: str, solver_name: str):
    solver_class = SOLVERS.get(problem_name, {}).get(solver_name)
    if not solver_class:
        print(f"Solver '{solver_name}' not found.")
        return
    print(f"Solver: {solver_name}")
    solver_instance = solver_class()
    print(f"Description: {solver_instance.description()}")
    dataset_name = questionary.select(
        "Choose a dataset to solve:", choices=list_datasets() + ["Go back"]
    ).ask()
    if dataset_name == "Go back":
        return
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return
    solutions = []
    for graph in tqdm(dataset):
        with graph as g:
            solutions.append(solver_instance.solve(g))
    solution_file = save_solver_solution(solver_name, solutions, dataset_name)
    print(f"Solutions saved to '{solution_file}'.")


if __name__ == "__main__":
    initialize_folders()

    while True:
        choice = questionary.select(
            "Choose an option:",
            choices=[
                "Explore datasets",
                "Explore graph creators",
                "Explore feature extractors",
                "Explore solvers",
                "Exit",
            ],
        ).ask()

        match choice:
            case "Explore datasets":
                explore_datasets()
            case "Explore graph creators":
                explore_graph_creators()
            case "Explore feature extractors":
                explore_feature_extractors()
            case "Explore solvers":
                explore_solvers()
            case "Exit":
                print("Exiting the program.")
                break
