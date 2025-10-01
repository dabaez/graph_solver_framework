import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

# This dataset must be modified to return PyG Data objects
from data.graph_dataset import GraphDataset

# Import the ported environment
from env import MaximumIndependentSetEnvPyG
from ppo.actor_critic import ActorCritic

# --- Import the ported PyG modules ---
from ppo.framework import ProxPolicyOptimFramework
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from ppo.storage import RolloutStorage

# --- PyG Imports ---
# Use the PyG DataLoader which handles graph batching automatically
from torch_geometric.loader import DataLoader

from .statistics import ResultCollector

# The argument parser remains unchanged.
parser = argparse.ArgumentParser()
parser.add_argument(
    "operation", type=str, help="Operation to perform.", choices=["train", "solve"]
)
parser.add_argument(
    "input",
    type=Path,
    action="store",
    help="Directory containing input graphs (to be solved/trained on).",
)
parser.add_argument(
    "output",
    type=Path,
    action="store",
    help="Folder in which the output (e.g. json containg statistics and solution will be stored, or trained weights)",
)

parser.add_argument(
    "--cuda_device",
    type=int,
    nargs="?",
    action="store",
    default=0,
    help="GPU device to use",
)
parser.add_argument(
    "--maximum_iterations_per_episode",
    type=int,
    nargs="?",
    action="store",
    default=32,
    help="Maximum iterations before the MDP timeouts.",
)
parser.add_argument(
    "--num_unrolling_iterations",
    type=int,
    nargs="?",
    action="store",
    default=32,
    help="Maximum number of unrolling iterations (how many stages we have per graph during training).",
)
parser.add_argument(
    "--num_environments_per_batch",
    type=int,
    nargs="?",
    action="store",
    default=32,
    help="Graph batch size during training",
)
parser.add_argument(
    "--gradient_step_batch_size",
    type=int,
    nargs="?",
    action="store",
    default=16,
    help="Batch size for gradient step",
)
parser.add_argument(
    "--gradient_steps_per_update",
    type=int,
    nargs="?",
    action="store",
    default=4,
    help="Number of gradient steps per update.",
)
parser.add_argument(
    "--diversity_reward_coefficient",
    type=float,
    nargs="?",
    action="store",
    default=0.1,
    help="Diversity reward coefficient.",
)
parser.add_argument(
    "--max_entropy_coefficient",
    type=float,
    nargs="?",
    action="store",
    default=0.1,
    help="Entropy coefficient.",
)
parser.add_argument(
    "--pretrained_weights",
    type=Path,
    nargs="?",
    action="store",
    help="Pre-trained weights to be used for solving/continuing training.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    nargs="?",
    action="store",
    default=10,
    help="How many solutions to sample (default in the paper: training=2, inference=10)",
)
parser.add_argument(
    "--num_updates",
    type=int,
    nargs="?",
    action="store",
    default=20000,
    help="How many PPO updates to do",
)
parser.add_argument(
    "--time_limit",
    type=int,
    nargs="?",
    action="store",
    default=600,
    help="Time limit in seconds",
)
parser.add_argument(
    "--noise_as_prob_maps",
    action="store_true",
    default=False,
    help="Use uniform noise instead of GNN output.",
)
parser.add_argument(
    "--training_graph_idx",
    type=int,
    nargs="?",
    action="store",
    help="On which graph index to continue training.",
)
parser.add_argument(
    "--max_nodes",
    type=int,
    nargs="?",
    action="store",
    help="If you have lots of graphs, the determiniation of maximum number of nodes takes some time. If this value is given, you can force-overwrite it to save time.",
)

args = parser.parse_args()

# All hyperparameter initializations remain the same.
# ... (rest of the initializations are identical)
device = torch.device(args.cuda_device)
base_data_dir = os.path.join(args.input)

# env
hamming_reward_coef = args.diversity_reward_coefficient

# actor critic
num_layers = 4
input_dim = 2
output_dim = 3
hidden_dim = 128

# optimization
init_lr = 1e-4
max_epi_t = args.maximum_iterations_per_episode
max_rollout_t = args.num_unrolling_iterations
max_update_t = args.num_updates

# ppo
gamma = 1.0
clip_value = 0.2
optim_num_samples = args.gradient_steps_per_update
critic_loss_coef = 0.5
reg_coef = args.max_entropy_coefficient
max_grad_norm = 0.5

# logging
vali_freq = 5
log_freq = 1

# main
rollout_batch_size = args.num_environments_per_batch
eval_batch_size = 1
optim_batch_size = args.gradient_step_batch_size
init_anneal_ratio = 1.0
max_anneal_t = -1
anneal_base = 0.0
train_num_samples = args.num_samples
eval_num_samples = args.num_samples

print("Variables initialized.")

# construct data loaders
# NOTE: The custom 'GraphDataset' must be updated to return
# torch_geometric.data.Data objects instead of DGL graphs.

print("Initializing dataset...")
# Assuming GraphDataset is adapted for PyG
dataset = GraphDataset(data_dir=args.input)
print("Initializing data loaders.")

# --- DGL to PyG DataLoader ---
# The PyG DataLoader handles batching of graphs automatically.
# We no longer need a custom collate_fn.
data_loaders = {
    "train": DataLoader(
        dataset,
        batch_size=rollout_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    ),
    "test": DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0
    ),
}

if not args.max_nodes:
    print("Determining maximum number of nodes over all graphs.")
    max_num_nodes = -1
    # Iterate through the PyG data loader
    for data in data_loaders["test"]:
        # For a PyG Batch object with batch_size=1, num_nodes gives the node count
        max_num_nodes = max(max_num_nodes, data.num_nodes)
else:
    print("Got maximum number of nodes from arguments!")
    max_num_nodes = args.max_nodes

print(f"Set max_num_nodes to {max_num_nodes}")

# construct environment using the ported PyG class
env = MaximumIndependentSetEnvPyG(
    max_epi_t=max_epi_t,
    max_num_nodes=max_num_nodes,
    hamming_reward_coef=hamming_reward_coef,
    device=device,
    time_limit=args.time_limit if args.operation == "solve" else None,
)

# Rollout storage is already ported and requires no change here
rollout = RolloutStorage(
    max_t=max_rollout_t, batch_size=rollout_batch_size, num_samples=train_num_samples
)

# ActorCritic and Frameworks are already ported and require no change here
actor_critic = ActorCritic(
    actor_class=PolicyGraphConvNet,
    critic_class=ValueGraphConvNet,
    max_num_nodes=max_num_nodes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    device=device,
)

framework = ProxPolicyOptimFramework(
    actor_critic=actor_critic,
    init_lr=init_lr,
    clip_value=clip_value,
    optim_num_samples=optim_num_samples,
    optim_batch_size=optim_batch_size,
    critic_loss_coef=critic_loss_coef,
    reg_coef=reg_coef,
    max_grad_norm=max_grad_norm,
    device=device,
)


# define evaluate function
def evaluate(actor_critic):
    actor_critic.eval()
    cum_cnt = 0
    cum_eval_sol = 0.0
    results = ResultCollector()

    # 'data' is now a PyG Batch object
    for g_idx, data in enumerate(data_loaders["test"]):
        g_name = dataset.graph_paths[g_idx]
        print(
            f"Evaluating graph {g_name} (graph {g_idx + 1}/{len(dataset.graph_paths)})"
        )
        collector = results.new_collector(g_name)
        collector.start_timer()
        collector.start_process_timer()
        # Use PyG properties for assertions
        assert data.num_graphs == 1

        # The DGL initializer is no longer needed
        # g.set_n_initializer(dgl.init.zero_initializer)

        ob = env.register(data, num_samples=eval_num_samples)
        while True:
            with torch.no_grad():
                # Pass the PyG 'data' object to the actor
                action = actor_critic.act(ob, data, random=args.noise_as_prob_maps)

            ob, reward, done, info = env.step(action)
            if torch.all(done).item():
                cum_eval_sol += info["sol"].max(dim=1)[0].sum().cpu()
                # Use PyG properties for counting
                cum_cnt += data.num_graphs

                best_sol_idx = info["sol"].max(dim=1)[1].cpu().item()
                best_sol_mis_size = info["sol"].max(dim=1)[0].cpu().item()
                best_sol = env.x[:, best_sol_idx].cpu().detach().numpy()

                collector.collect_result(np.flatnonzero(best_sol))
                collector.stop_timer()
                print(f"Done! Found MIS of size {best_sol_mis_size}")

                break

    actor_critic.train()
    avg_eval_sol = cum_eval_sol / cum_cnt
    results.finalize(args.output / "results.json")

    return avg_eval_sol


def train():
    for update_t in range(max_update_t):
        if update_t == 0 or torch.all(done).item():
            try:
                data = next(train_data_iter)
                g_idx += 1
            except:
                train_data_iter = iter(data_loaders["train"])
                data = next(train_data_iter)
                g_idx = 0

                if args.training_graph_idx:
                    print(
                        f"Continuing training at graph index {args.training_graph_idx}"
                    )
                    while g_idx < args.training_graph_idx:
                        data = next(train_data_iter)
                        g_idx += 1

            # DGL initializer is removed
            ob = env.register(data, num_samples=train_num_samples)
            # Use the ported 'insert_ob_and_data' method
            rollout.insert_ob_and_data(ob, data)

        for step_t in range(max_rollout_t):
            with torch.no_grad():
                # Pass the PyG 'data' object
                (
                    action,
                    action_log_prob,
                    value_pred,
                ) = actor_critic.act_and_crit(ob, data)

            ob, reward, done, info = env.step(action)

            rollout.insert_tensors(
                ob, action, action_log_prob, value_pred, reward, done
            )

            if torch.all(done).item():
                avg_sol = info["sol"].max(dim=1)[0].mean().cpu()
                break

        rollout.compute_rets_and_advantages(gamma)
        actor_loss, critic_loss, entropy_loss = framework.update(rollout)

        if (update_t + 1) % log_freq == 0:
            print("update_t: {:05d}".format(update_t + 1))
            print("train stats...")
            print(
                "sol: {:.4f}, "
                "actor_loss: {:.4f}, "
                "critic_loss: {:.4f}, "
                "entropy: {:.4f}".format(
                    avg_sol, actor_loss.item(), critic_loss.item(), entropy_loss.item()
                )
            )
            # Use PyG property for logging
            print(f"current graph = {g_idx} (batch size = {data.num_graphs})")

        if update_t % 300 == 0 or update_t == max_update_t:
            print("saving intermediate results...")
            torch.save(
                actor_critic.actor_net.state_dict(),
                args.output / f"{update_t}_{g_idx}_{time.monotonic()}_actornet.torch",
            )
            torch.save(
                actor_critic.critic_net.state_dict(),
                args.output / f"{update_t}_{g_idx}_{time.monotonic()}_criticnet.torch",
            )

    print("Training finished, writing state")
    torch.save(actor_critic.actor_net.state_dict(), args.output / "actornet.torch")
    torch.save(actor_critic.critic_net.state_dict(), args.output / "criticnet.torch")


# Loading weights and running train/solve logic remains the same
if args.pretrained_weights:
    print("Loading pretrained weights")
    actor_critic.actor_net.load_state_dict(
        torch.load(args.pretrained_weights / "actornet.torch")
    )
    actor_critic.critic_net.load_state_dict(
        torch.load(args.pretrained_weights / "criticnet.torch")
    )
    print("Weights loaded")

if args.operation == "train":
    print("Starting training")
    train()
    print("Training finished, exiting.")
else:
    print("Starting evaluation.")
    res = evaluate(actor_critic)
    print("Evaluation done.")
