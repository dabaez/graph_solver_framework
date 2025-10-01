import torch
from config import (
    cuda_devs,
    diversity_reward_coefficient,
    gradient_step_batch_size,
    gradient_steps_per_update,
    max_entropy_coefficient,
    maximum_iterations_per_episode,
    num_environments_per_batch,
    num_samples,
    num_unrolling_iterations,
    num_updates,
)

from framework.core.graph import Dataset


def train(dataset: Dataset):
    device = torch.device(cuda_devs)

    # env
    hamming_reward_coefficient = diversity_reward_coefficient

    # actor critic
    num_layers = 4
    input_dim = 2
    output_dim = 3
    hidden_dim = 128

    # optimization
    init_lr = 1e-4
    max_epi_t = maximum_iterations_per_episode
    max_rollout_t = num_unrolling_iterations
    max_update_t = num_updates

    # ppo
    gamma = 1.0
    clip_value = 0.2
    optim_num_samples = gradient_steps_per_update
    critic_loss_coef = 0.5
    reg_coef = max_entropy_coefficient
    max_grad_norm = 0.5

    # logging
    vali_freq = 5
    log_freq = 1

    # main
    rollout_batch_size = num_environments_per_batch
    eval_batch_size = 1
    optim_batch_size = gradient_step_batch_size
    init_anneal_ratio = 1.0
    max_anneal_t = -1
    anneal_base = 0.0
    train_num_samples = num_samples
    eval_num_samples = num_samples

    data_loader
