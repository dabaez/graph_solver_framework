import time

import torch
from torch_geometric.nn.pool import global_add_pool
from torch_scatter import scatter_add


class MaximumIndependentSetEnvPyG(object):
    def __init__(
        self, max_epi_t, max_num_nodes, hamming_reward_coef, device, time_limit
    ):
        self.max_epi_t = max_epi_t
        self.max_num_nodes = max_num_nodes
        self.hamming_reward_coef = hamming_reward_coef
        self.device = device
        self.time_limit = time_limit
        self.start_time = None

        if self.time_limit is None:
            print(
                "disabled time limit for MIS, probably due to training instead of solving."
            )

    def step(self, action):
        if self.time_limit is not None and self.start_time is None:
            self.start_time = time.monotonic()

        reward, sol, done = self._take_action(action)

        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}

        return ob, reward, done, info

    def _take_action(self, action):
        undecided = self.x == 2
        self.x[undecided] = action[undecided]
        self.t += 1

        x1 = self.x == 1

        edge_src, edge_dst = self.data.edge_index
        # Propagate "1"s from source nodes in the solution to destination nodes
        x1_float = x1.float()
        # Ensure x1_float is properly shaped for scatter operation
        if x1_float.dim() == 1:
            x1_float = x1_float.unsqueeze(1)

        # Sum messages at destination nodes
        messages = x1_float[edge_src]
        x1_deg = scatter_add(messages, edge_dst, dim=0, dim_size=self.data.num_nodes)

        ## forgive clashing
        clashed = x1 & (x1_deg > 0)
        self.x[clashed] = 2
        x1_deg[clashed] = 0

        # graph clean up
        still_undecided = self.x == 2
        self.x[still_undecided & (x1_deg > 0)] = 0

        # fill timeout with zeros
        still_undecided = self.x == 2
        timeout = self.t == self.max_epi_t

        if (
            self.time_limit is not None
            and self.start_time is not None
            and time.monotonic() - self.start_time > self.time_limit
        ):
            timeout = True
            print("Time-based timeout! Setting all undecided vertices to 0.")

        self.x[still_undecided & timeout] = 0

        done = self._check_done()
        self.epi_t[~done] += 1

        # compute reward and solution
        node_sol = (self.x == 1).float()

        next_sol = global_add_pool(node_sol, self.data.batch)

        reward = next_sol - self.sol

        if self.hamming_reward_coef > 0.0 and self.num_samples == 2:
            xl, xr = self.x.split(1, dim=1)
            undecidedl, undecidedr = undecided.split(1, dim=1)
            hamming_d = torch.abs(xl.float() - xr.float())
            hamming_d[(xl == 2) | (xr == 2)] = 0.0
            hamming_d[~undecidedl & ~undecidedr] = 0.0

            hamming_reward = global_add_pool(hamming_d, self.data.batch).expand_as(
                reward
            )
            reward += self.hamming_reward_coef * hamming_reward

        reward /= self.max_num_nodes

        return reward, next_sol, done

    def _check_done(self):
        undecided = (self.x == 2).float()

        num_undecided = global_add_pool(undecided, self.data.batch)
        done = num_undecided == 0

        return done

    def _build_ob(self):
        ob_x = self.x.unsqueeze(2).float()
        # self.t is already [num_nodes, num_samples], so we need to unsqueeze correctly
        ob_t = self.t.unsqueeze(2).float() / self.max_epi_t
        ob = torch.cat([ob_x, ob_t], dim=2)
        return ob

    def register(self, data, num_samples=1):
        # The input 'data' is now expected to be a PyG Batch object
        self.data = data.to(self.device)
        self.num_samples = num_samples

        self.batch_num_nodes = torch.bincount(self.data.batch).to(self.device)
        num_nodes = self.data.num_nodes
        batch_size = self.data.num_graphs

        self.x = torch.full(
            (num_nodes, num_samples), 2, dtype=torch.long, device=self.device
        )
        self.t = torch.zeros(
            num_nodes, num_samples, dtype=torch.long, device=self.device
        )

        ob = self._build_ob()

        self.sol = torch.zeros(batch_size, num_samples, device=self.device)
        self.epi_t = torch.zeros(batch_size, num_samples, device=self.device)

        if self.time_limit is not None:
            self.start_time = time.monotonic()

        return ob
