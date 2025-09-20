import argparse
import os
import random
import time
from pathlib import Path

import logzero
import numpy as np
import questionary
import torch
import torch_geometric.transforms as T
from logzero import logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from framework.core.graph import Dataset
from framework.experiment.utils import list_datasets, load_dataset

from .model import HindsightLoss
from .utils import _load_model


def convert_dataset_to_pyg(dataset: Dataset) -> list[Data]:
    pyg_dataset = []
    for graph in dataset:
        G = graph.graph_object.copy()
        pyg_data = from_networkx(G)

        if "labels" not in graph.features:
            raise ValueError("Graph must contain 'labels' feature.")

        node_labels = graph.features["labels"]

        if not isinstance(node_labels, np.ndarray):
            raise TypeError("Expected 'labels' to be a NumPy array.")

        labels_tensor = torch.from_numpy(node_labels).long()

        if pyg_data.num_nodes != labels_tensor.shape[0]:
            raise ValueError("Inconsistent number of nodes and features/labels.")

        pyg_data.x = torch.ones((pyg_data.num_nodes, 1), dtype=torch.float)
        pyg_data.y = labels_tensor

        pyg_dataset.append(pyg_data)

    return pyg_dataset


def train(
    self_loop: bool,
    cuda_devices: list[int],
    model_prob_maps: int,
    input: Dataset,
    output_path: Path,
    lr: float,
    epochs: int,
    pretrained_weights: Path | None,
):
    cuda = bool(cuda_devices)
    prob_maps = model_prob_maps

    if cuda and torch.cuda.is_available():
        device_id = cuda_devices[0]
        if len(cuda_devices) > 1:
            logger.warning(
                f"More than one cuda devices was provided, using {device_id}"
            )
        cuda_dev = torch.device(f"cuda:{device_id}")
        logger.info(f"Using CUDA device: {cuda_dev}")
    else:
        cuda_dev = torch.device("cpu")

    graphs = convert_dataset_to_pyg(input)
    random.shuffle(graphs)

    transform_list: list = [T.RemoveSelfLoops()]
    if self_loop:
        transform_list.append(T.AddSelfLoops())
    transform = T.Compose(transform_list)

    processed_graphs = [transform(g) for g in graphs]

    logger.info(f"Number of training graphs: {len(processed_graphs)}")

    train_loader = DataLoader(
        processed_graphs,
        batch_size=32,
        shuffle=True,
    )

    model = _load_model(
        prob_maps=prob_maps, weight_file=pretrained_weights, cuda_dev=cuda_dev
    )
    model.to(cuda_dev)

    loss_fcn = HindsightLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{epochs - 1}")

        model.train()

        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(cuda_dev)

            output = model(batch.x, batch.edge_index)
            loss = loss_fcn(output, batch.y)

            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch} finished. Average loss {avg_epoch_loss:.4f}")

        torch.save(
            model.state_dict(),
            output_path
            / f"{int(time.time())}_intermediate_model{prob_maps}_{epoch}_{avg_epoch_loss:.2f}.pt",
        )

    logger.info(
        f"Final: Average Epoch Loss: {avg_epoch_loss:.2f}, Last Loss: {loss.item():.2f}"
    )
    torch.save(
        model.state_dict(),
        output_path / f"{int(time.time())}_final_model{prob_maps}.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the DGL Tree Search model.")

    parser.add_argument(
        "--input_dataset",
        type=str,
        help="Name of the dataset to train with (if not chosen, it will be asked for)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output directory (if not given, it will be asked for)",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=Path,
        help="Path to the pretrained weights file",
    )
    parser.add_argument(
        "--self_loops",
        action="store_true",
        default=False,
        help="Whether to add self-loops to the graphs",
    )
    parser.add_argument(
        "--cuda_devices",
        nargs="*",
        type=int,
        default=[],
        help="List of CUDA devices to use",
    )
    parser.add_argument(
        "--model_prob_maps",
        type=int,
        default=32,
        help="Number of probability maps to train the model for.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        action="store",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of logging (DEBUG/INFO/WARNING/ERROR)",
    )

    args = parser.parse_args()

    if args.loglevel == "DEBUG":
        logzero.loglevel(logzero.DEBUG)
    elif args.loglevel == "INFO":
        logzero.loglevel(logzero.INFO)
    elif args.loglevel == "WARNING":
        logzero.loglevel(logzero.WARNING)
    elif args.loglevel == "ERROR":
        logzero.loglevel(logzero.ERROR)
    else:
        print(f"Unknown loglevel {args.loglevel}, ignoring.")

    if args.input_dataset is None:
        all_datasets = list_datasets()
        args.input_dataset = questionary.select(
            "Select a dataset to train with:", choices=all_datasets
        ).ask()

    if args.output_dir is None:
        args.output_dir = Path(questionary.path("Select an output directory:").ask())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(
        args.self_loops,
        args.cuda_devices,
        args.model_prob_maps,
        load_dataset(args.input_dataset),
        args.output_dir,
        args.lr,
        args.epochs,
        args.pretrained_weights,
    )
