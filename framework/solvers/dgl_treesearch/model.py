from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int,
        activation: Callable[[Tensor], Tensor],
        dropout: float,
    ):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.activation = activation

        self.convs.append(GCNConv(in_feats, n_hidden))

        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(n_hidden, n_hidden))

        self.convs.append(GCNConv(n_hidden, n_classes))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
        h = self.convs[-1](h, edge_index)
        return torch.sigmoid(h)


class HindsightLoss(nn.Module):
    def __init__(self):
        super(HindsightLoss, self).__init__()
        self.ce_func = nn.BCELoss(reduction="none")

    def forward(self, output, labels):
        probmaps = output.shape[1]
        _labels = torch.unsqueeze(labels, 0)
        _labels = _labels.float().repeat(probmaps, 1)
        output = output.permute(1, 0)

        loss = torch.min(torch.mean(self.ce_func(output, _labels), dim=1))
        return loss
