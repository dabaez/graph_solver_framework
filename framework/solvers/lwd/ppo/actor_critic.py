import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import subgraph


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor_class,  # Expects the PyG-compatible PolicyGraphConvNet
        critic_class,  # Expects the PyG-compatible ValueGraphConvNet
        max_num_nodes,
        hidden_dim,
        num_layers,
        device,
    ):
        super(ActorCritic, self).__init__()
        # These now instantiate the PyG versions of the networks
        self.actor_net = actor_class(2, hidden_dim, 3, num_layers)
        self.critic_net = critic_class(2, hidden_dim, 1, num_layers)
        self.device = device
        self.to(device)
        self.max_num_nodes = max_num_nodes

    def get_masks_idxs_subg_h(self, ob, data):
        # This part is mostly tensor manipulation and remains the same.
        # It identifies undecided nodes to form a subgraph for processing.
        node_mask = ob.select(2, 0).long() == 2
        flatten_node_idxs = node_mask.view(-1).nonzero().squeeze(1)

        subg_mask = node_mask.any(dim=1)
        flatten_subg_idxs = subg_mask.nonzero().squeeze(1)

        subg_node_mask = node_mask.index_select(0, flatten_subg_idxs)
        flatten_subg_node_idxs = subg_node_mask.view(-1).nonzero().squeeze(1)

        # --- DGL to PyG Subgraph Extraction ---
        # Instead of g.subgraph, we use pyg.utils.subgraph
        # It returns a new edge_index with remapped node indices.
        subg_edge_index, _ = subgraph(
            flatten_subg_idxs,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        # We create a new lightweight Data object for the subgraph
        subg_data = Data(
            edge_index=subg_edge_index, num_nodes=flatten_subg_idxs.size(0)
        )
        subg_data = subg_data.to(self.device)

        # Feature selection logic remains the same
        h = self._build_h(ob).index_select(0, flatten_subg_idxs)

        return (
            (node_mask, subg_mask, subg_node_mask),
            (flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs),
            subg_data,  # Return the PyG data object for the subgraph
            h,
        )

    def act(self, ob, data, random=False):
        num_nodes, batch_size = ob.size(0), ob.size(1)

        masks, idxs, subg_data, h = self.get_masks_idxs_subg_h(ob, data)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # The actor_net now takes the PyG subg_data object
        logits = (
            self.actor_net(
                h,
                subg_data,  # Pass the PyG Data object
            )
            .view(-1, 3)
            .index_select(0, flatten_subg_node_idxs)
        )

        if random:
            min_elem = torch.min(logits)
            max_elem = torch.max(logits)
            logits = min_elem + torch.rand_like(logits) * (max_elem - min_elem)

        # Action sampling logic remains the same
        action = torch.zeros(
            num_nodes * batch_size, dtype=torch.long, device=self.device
        )
        m = Categorical(logits=logits.view(-1, logits.size(-1)))
        action[flatten_node_idxs] = m.sample()
        action = action.view(-1, batch_size)

        return action

    def act_and_crit(self, ob, data):
        num_nodes, batch_size = ob.size(0), ob.size(1)

        masks, idxs, subg_data, h = self.get_masks_idxs_subg_h(ob, data)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # compute logits to get action
        logits = (
            self.actor_net(
                h,
                subg_data,  # Pass the PyG Data object
            )
            .view(-1, 3)
            .index_select(0, flatten_subg_node_idxs)
        )

        m = Categorical(logits=logits)
        action = torch.zeros(
            num_nodes * batch_size, dtype=torch.long, device=self.device
        )
        action[flatten_node_idxs] = m.sample()

        action_log_probs = torch.zeros(num_nodes * batch_size, device=self.device)
        action_log_probs[flatten_node_idxs] = m.log_prob(
            action.index_select(0, flatten_node_idxs)
        )

        action = action.view(-1, batch_size)
        action_log_probs = action_log_probs.view(-1, batch_size)

        # compute value predicted by critic
        node_value_preds = torch.zeros(num_nodes * batch_size, device=self.device)

        node_value_preds[flatten_node_idxs] = (
            self.critic_net(
                h,
                subg_data,  # Pass the PyG Data object
            )
            .view(-1)
            .index_select(0, flatten_subg_node_idxs)
        )

        # --- DGL to PyG Graph-level Aggregation ---
        # Replace dgl.sum_nodes with pyg.nn.global_add_pool
        node_value_preds_reshaped = node_value_preds.view(num_nodes, batch_size)
        value_pred = (
            global_add_pool(node_value_preds_reshaped, data.batch) / self.max_num_nodes
        )

        return action, action_log_probs, value_pred

    def evaluate_batch(self, ob, data, action):
        num_nodes, batch_size = ob.size(0), ob.size(1)
        masks, idxs, subg_data, h = self.get_masks_idxs_subg_h(ob, data)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # compute logits to get action
        logits = (
            self.actor_net(
                h,
                subg_data,  # Pass the PyG Data object
            )
            .view(-1, 3)
            .index_select(0, flatten_subg_node_idxs)
        )

        m = Categorical(logits=logits)

        # compute log probability of actions per node
        action_log_probs = torch.zeros(num_nodes * batch_size, device=self.device)

        action_log_probs[flatten_node_idxs] = m.log_prob(
            action.reshape(-1).index_select(0, flatten_node_idxs)
        )
        action_log_probs = action_log_probs.view(-1, batch_size)

        node_entropies = -torch.sum(
            torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
        )
        avg_entropy = node_entropies.mean()

        # compute value predicted by critic
        node_value_preds = torch.zeros(num_nodes * batch_size, device=self.device)

        node_value_preds[flatten_node_idxs] = (
            self.critic_net(
                h,
                subg_data,  # Pass the PyG Data object
            )
            .view(-1)
            .index_select(0, flatten_subg_node_idxs)
        )

        # --- DGL to PyG Graph-level Aggregation ---
        node_value_preds_reshaped = node_value_preds.view(num_nodes, batch_size)
        value_preds = (
            global_add_pool(node_value_preds_reshaped, data.batch) / self.max_num_nodes
        )

        return action_log_probs, avg_entropy, value_preds, node_mask

    def _build_h(self, ob):
        # This helper function is graph-agnostic and remains unchanged.
        ob_t = ob.select(2, 1).unsqueeze(2)
        return torch.cat([ob_t, torch.ones_like(ob_t)], dim=2)
