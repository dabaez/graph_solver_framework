from dataclasses import dataclass

from framework.core.solver import MaximumIndependentSet


@dataclass
class Solution:
    mis: MaximumIndependentSet
    time: float
