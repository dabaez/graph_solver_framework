import random
import time

import torch
import torch_geometric.transforms as T
from logzero import logger

from .model import HindsightLoss
from .utils import _load_model


def train(self_loop, cuda_devices, model_prob_maps, input, output, lr, epochs):
    cuda = bool(cuda_devices)
    prob_maps = model_prob_maps

    if cuda:
        cuda_dev = cuda_devices[0]
        if len(cuda_devices) > 1:
            logger.warning(f"More than one cuda devices was provided, using {cuda_dev}")
    else:
        cuda_dev = None

    training_graphs = []

    graphs = torch.load(input / "graphs.pt")
    random.shuffle(graphs)

    for graph in graphs:
        if cuda:
            graph = graph.to(cuda_dev)

        if self_loop:
            transforms = T.Compose([T.RemoveSelfLoops(), T.AddSelfLoops()])
            graph = transforms(graph)
        else:
            transforms = T.RemoveSelfLoops()
            graph = transforms(graph)

        training_graphs.append(graph)

    logger.info(f"Number of training graphs: {len(training_graphs)}")

    model = _load_model(prob_maps, cuda_dev=cuda_dev)
    loss_fcn = HindsightLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    num_epochs = epochs
    status_update_every = max(1, int(0.1 * len(training_graphs)))

    for epoch in range(num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        epoch_losses = []
        for gidx, graph in enumerate(training_graphs):
            features = graph.x
            labels = graph.y
            model.train()
            output = model(graph, features)
            loss = loss_fcn(output, labels)
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gidx % status_update_every == 0:
                logger.info(
                    f"Epoch {epoch}, Graph {gidx}/{len(training_graphs)}: Average Epoch Loss: {sum(epoch_losses) / len(epoch_losses):.2f}, Last Training Loss: {loss.item():.2f}"
                )
        torch.save(
            model.state_dict(),
            output
            / f"{int(time.time())}_intermediate_model{prob_maps}_{epoch}_{sum(epoch_losses) / len(epoch_losses):.2f}.pt",
        )

    logger.info(
        f"Final: Average Epoch Loss: {sum(epoch_losses) / len(epoch_losses):.2f}, Last Loss: {loss.item():.2f}"
    )
    torch.save(
        model.state_dict(), output / f"{int(time.time())}_final_model{prob_maps}.pt"
    )
