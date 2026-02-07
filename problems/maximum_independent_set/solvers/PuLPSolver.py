import networkx as nx
import pulp

from framework.core.registries import register_solver
from problems.maximum_independent_set.problem import MaximumIndependentSetSolution


@register_solver("MaximumIndependentSetProblem", "PuLPSolver")
class PuLPSolver:
    def description(self) -> str:
        return "A solver that uses PuLP to solve the Maximum Independent Set problem via Integer Linear Programming."

    def solve(self, graph: nx.Graph) -> MaximumIndependentSetSolution:
        prob = pulp.LpProblem("Maximum_Independent_Set", pulp.LpMaximize)

        node_vars = {
            str(node): pulp.LpVariable(f"x_{node}", cat="Binary")
            for node in graph.nodes()
        }
        prob += pulp.lpSum(node_vars[str(node)] for node in graph.nodes()), "Objective"

        for u, v in graph.edges():
            prob += node_vars[str(u)] + node_vars[str(v)] <= 1, f"Edge_{u}_{v}"

        solver = pulp.PULP_CBC_CMD(msg=False)
        solver.timeLimit = 30
        prob.solve(solver)
        independent_set = [
            str(node) for node in graph.nodes() if pulp.value(node_vars[str(node)]) == 1
        ]
        return MaximumIndependentSetSolution(
            set=independent_set, time=prob.solutionTime
        )
