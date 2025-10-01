import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class GraphConvPyG(MessagePassing):
    """
    This layer is a PyTorch Geometric reimplementation of the original DGL-based GraphConv layer.
    It replicates the core logic:
    1. Symmetric degree normalization.
    2. Feature aggregation.
    3. An optional "jump" connection (concatenating input features with aggregated ones).
    4. A final linear transformation and activation.
    """

    def __init__(self, in_feats, out_feats, jump=True, activation=None):
        # "add" aggregation is equivalent to DGL's sum.
        super(GraphConvPyG, self).__init__(aggr="add")
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._jump = jump
        self._activation = activation

        # The final linear transformation is applied after aggregation and jump connection.
        if jump:
            self.weight = nn.Parameter(torch.Tensor(2 * in_feats, out_feats))
        else:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        self.bias = nn.Parameter(torch.Tensor(out_feats))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, feat, edge_index):
        # Store original features for the jump connection
        if self._jump:
            _feat = feat

        # 1. & 2. Compute normalization and propagate messages
        # edge_index, _ = add_self_loops(edge_index, num_nodes=feat.size(0)) # Optional: Add self-loops

        # Calculate degrees
        row, col = edge_index
        deg = degree(col, feat.size(0), dtype=feat.dtype)

        # Compute normalization constant D^(-0.5)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagation
        # The 'propagate' method calls message(), aggregate(), and update()
        rst = self.propagate(edge_index, x=feat, norm=norm)

        # 3. Apply jump connection if specified
        if self._jump:
            rst = torch.cat([rst, _feat], dim=-1)

        # 4. Apply final linear transformation
        rst = torch.matmul(rst, self.weight)

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def message(self, x_j, norm):
        # x_j has shape [E, in_feats]
        # Multiply source node features 'x_j' by the normalization term 'norm'.
        return norm.view(-1, 1) * x_j


class PolicyGraphConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PolicyGraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        # Use the new PyG-compatible layer
        self.layers.append(GraphConvPyG(input_dim, hidden_dim, activation=F.relu))
        for i in range(num_layers - 1):
            self.layers.append(GraphConvPyG(hidden_dim, hidden_dim, activation=F.relu))

        self.layers.append(GraphConvPyG(hidden_dim, output_dim, activation=None))

        # This important initialization detail is preserved
        with torch.no_grad():
            self.layers[-1].bias[2].add_(3.0)  # type: ignore

    def forward(self, h, data):
        # The 'data' object from the environment now provides the edge_index
        edge_index = data.edge_index
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)

        return h


class ValueGraphConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ValueGraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        # Use the new PyG-compatible layer
        self.layers.append(GraphConvPyG(input_dim, hidden_dim, activation=F.relu))
        for i in range(num_layers - 1):
            self.layers.append(GraphConvPyG(hidden_dim, hidden_dim, activation=F.relu))

        self.layers.append(GraphConvPyG(hidden_dim, output_dim, activation=None))

    def forward(self, h, data):
        # The 'data' object from the environment now provides the edge_index
        edge_index = data.edge_index
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)

        return h
