import logzero

from framework.core.graph import FrameworkGraph
from framework.core.registries import register_solver
from framework.core.solver import Solution

from .config import (
    cuda_devs,
    local_search,
    logs,
    max_prob_maps,
    model_prob_maps,
    noise_as_prob_maps,
    pretrained_weights,
    queue_pruning,
    reduction,
    self_loop,
    thread_count,
    time_budget,
    weighted_queue_pop,
)
from .solve import solve


@register_solver("PYGTreeSearchSolver")
class PYGTreeSearchSolver:
    def description(self) -> str:
        return "A solver that uses a tree search algorithm together with a neural network from PyTorch Geometric to find a maximum independent set."

    def solve(self, graph: FrameworkGraph) -> Solution:
        if not logs:
            logzero.loglevel(logzero.ERROR)
        solution = solve(
            self_loop=self_loop,
            threadcount=thread_count,
            max_prob_maps=max_prob_maps,
            model_prob_maps=model_prob_maps,
            cuda_devs=cuda_devs,
            time_budget=time_budget,
            reduction=reduction,
            local_search=local_search,
            queue_pruning=queue_pruning,
            noise_as_prob_maps=noise_as_prob_maps,
            weighted_queue_pop=weighted_queue_pop,
            pretrained_weights=pretrained_weights,
            input=graph,
        )
        return solution
