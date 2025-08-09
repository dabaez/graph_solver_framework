import questionary

import framework.dataset_creators  # noqa: F401
import framework.feature_extractors  # noqa: F401
import framework.solvers  # noqa: F401
from framework.core.registries import (
    DATASET_CREATORS,
    FEATURE_EXTRACTORS,
    SOLVERS,
)

from .config import DATA_FOLDER, DATASETS_FOLDER, SOLUTIONS_FOLDER
from .utils import (
    CalculatedFeaturesFromDataset,
    CalculatedFeaturesPercentageFromDataset,
    create_folder_if_not_exists,
    dataset_exists,
    delete_dataset,
    list_datasets,
    list_registered_dataset_creators,
    list_registered_dataset_feature_extractors,
    list_registered_dataset_solvers,
    load_dataset,
    merge_datasets,
    rename_dataset,
    save_dataset,
    save_dataset_with_name,
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
        print("Dataset has the following calculated features:")
        for feature, percentage in calculated_features.items():
            print(
                f"{feature}: {percentage:.2f}% of graphs have this feature calculated."
            )
    else:
        print("No calculated features found in this dataset.")
    chosen_option = questionary.select(
        "Choose an option:",
        choices=[
            "Explore graphs in this dataset",
            "Rename this dataset",
            "Delete this dataset",
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
                    print(graph.graph_object.edges())
            except (ValueError, IndexError):
                print("Invalid graph index. Please try again.")
    elif chosen_option == "Rename this dataset":
        while True:
            new_name = questionary.text(
                "Enter a new name for the dataset (or leave empty to go back):"
            ).ask()
            if not new_name:
                return
            if dataset_exists(new_name):
                print(
                    f"Dataset '{new_name}' already exists. Please choose a different name."
                )
            else:
                break
        rename_dataset(dataset_name, new_name)
        print(f"Dataset renamed to '{new_name}'.")
        explore_dataset(new_name)
    elif chosen_option == "Delete this dataset":
        if questionary.confirm(
            f"Are you sure you want to delete the dataset '{dataset_name}'?"
        ).ask():
            delete_dataset(dataset_name)
            print(f"Dataset '{dataset_name}' deleted.")


def explore_dataset_creators():
    creators = list_registered_dataset_creators()
    if not creators:
        print("No dataset creators available.")
        return
    chosen_dataset_creator = questionary.select(
        "Choose a dataset creator to explore:", choices=[*creators, "Go back"]
    ).ask()
    if chosen_dataset_creator == "Go back":
        return
    explore_dataset_creator(chosen_dataset_creator)


def explore_dataset_creator(dataset_creator_name: str):
    creator_class = DATASET_CREATORS.get(dataset_creator_name)
    if not creator_class:
        print(f"Dataset creator '{dataset_creator_name}' not found.")
        return
    print(f"Dataset Creator: {dataset_creator_name}")
    creator_instance = creator_class()
    print(f"Description: {creator_instance.description()}")
    required_params = creator_instance.required_parameters()
    if required_params:
        print("Required parameters:")
        for param in required_params:
            print(f"- {param.name}: {param.description}")
    else:
        print("This dataset creator does not require any parameters.")

    create_a_dataset = questionary.confirm(
        "Do you want to create a dataset using this creator?"
    ).ask()

    if create_a_dataset:
        parameters = {param.name: "" for param in required_params}
        while True:
            for param in required_params:
                value = questionary.text(
                    f"{param.name}:", default=parameters[param.name]
                ).ask()
                parameters[param.name] = value

            if creator_instance.validate_parameters(parameters):
                dataset = creator_instance.create_dataset(parameters)
                dataset_file = save_dataset(dataset)
                print(f"Dataset created and saved as '{dataset_file}'.")
                break
            else:
                print("Invalid parameters provided.")
                if not questionary.confirm("Do you want to try again?").ask():
                    break


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
    overwrite__features = True
    if any(feature in feature_names for feature in present_features):
        overwrite__features = questionary.confirm(
            "Some features are already present in the dataset. Do you want to overwrite them? 'Y' for overwrite, 'N' to skip those features."
        ).ask()
    for graph in dataset:
        calculated_features = extractor_instance.extract_features(graph)
        for feature in calculated_features:
            graph.add_feature(feature, overwrite=overwrite__features)
    save_dataset_with_name(dataset, dataset_name)
    print(f"Features extracted and added to the dataset '{dataset_name}'.")


def explore_solvers():
    solvers = list_registered_dataset_solvers()
    if not solvers:
        print("No dataset solvers available.")
        return
    chosen_solver = questionary.select(
        "Choose a dataset solver to explore:", choices=[*solvers, "Go back"]
    ).ask()
    if chosen_solver == "Go back":
        return
    explore_solver(chosen_solver)


def explore_solver(solver_name: str):
    solver_class = SOLVERS.get(solver_name)
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
    solutions = [solver_instance.solve(graph) for graph in dataset]
    solution_file = save_solver_solution(solver_name, solutions, dataset_name)
    print(f"Solutions saved to '{solution_file}'.")


if __name__ == "__main__":
    initialize_folders()

    while True:
        choice = questionary.select(
            "Choose an option:",
            choices=[
                "Explore datasets",
                "Explore dataset creators",
                "Explore feature extractors",
                "Explore solvers",
                "Exit",
            ],
        ).ask()

        match choice:
            case "Explore datasets":
                explore_datasets()
            case "Explore dataset creators":
                explore_dataset_creators()
            case "Explore feature extractors":
                explore_feature_extractors()
            case "Explore solvers":
                explore_solvers()
            case "Exit":
                print("Exiting the program.")
                break
