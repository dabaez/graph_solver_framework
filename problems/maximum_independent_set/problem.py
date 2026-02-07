from dataclasses import dataclass

from networkx import Graph

from framework.core.registries import register_problem


@dataclass
class MaximumIndependentSetSolution:
    set: list[str]
    time: float

    def __dict__(self) -> dict[str, str]:
        return {
            "Solution Size": str(len(self.set)),
            "Time": str(self.time),
            "Set": ", ".join(self.set),
        }


@register_problem("MaximumIndependentSetProblem")
class MaximumIndependentSetProblem:
    def name(self) -> str:
        return "Maximum Independent Set"

    def description(self) -> str:
        return "Find the largest independent set of a graph."

    def is_valid(self, graph: Graph, solution: MaximumIndependentSetSolution) -> bool:
        # Check if the solution is an independent set
        for node in solution.set:
            for neighbor in graph.neighbors(node):
                if neighbor in solution.set:
                    return False
        return True

    def is_solution_worse(
        self,
        solution_a: MaximumIndependentSetSolution,
        solution_b: MaximumIndependentSetSolution,
    ) -> bool:
        # A solution is worse if it has a smaller independent set
        if len(solution_a.set) < len(solution_b.set):
            return True
        if len(solution_a.set) > len(solution_b.set):
            return False
        # If sizes are equal, compare by time taken (with a tolerance of 5 seconds)
        if solution_a.time > solution_b.time + 5:
            return True
        return False
