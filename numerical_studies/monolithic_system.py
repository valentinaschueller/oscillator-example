from abc import ABC, abstractmethod

import numpy as np
from timestepping import TimesteppingMethod


class MonolithicSystem(ABC):
    def __init__(self, k1, k2, k12, m1, m2):
        # for the formulation: u'' = A_second_order * u:
        self.A_second_order = np.array(
            [[-(k1 + k12) / m1, k12 / m1], [k12 / m2, -(k2 + k12) / m2]]
        )
        # for the formulation: Mu'' + Ku = 0:
        self.M = np.array([[m1, 0.0], [0.0, m2]])
        self.K = np.array([[(k1 + k12), -k12], [-k12, (k2 + k12)]])
        # for the formulation: y' = A_first_order * y:
        self.A_first_order = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-(k1 + k12) / m1, k12 / m1, 0.0, 0.0],
                [k12 / m2, -(k2 + k12) / m2, 0.0, 0.0],
            ]
        )

    def second_order_force(self, t, t_lower):
        del t, t_lower  # no time-dependent force for this system
        return np.array([0.0, 0.0])

    def first_order_force(self, t, t_lower):
        del t, t_lower  # no time-dependent force for this system
        return np.array([0.0, 0.0, 0.0, 0.0])

    @abstractmethod
    def _initial_conditions(self):
        pass

    def numerical_solution(self, t_end: float, N: int, solver: TimesteppingMethod):
        dt = t_end / N
        try:
            result = np.zeros((6, N + 1))
            result[:, 0] = self._initial_conditions()
            solver.integrate(dt, N, result)
        except ValueError:
            result = np.zeros((4, N + 1))
            result[:, 0] = self._initial_conditions()[:4]
            solver.integrate(dt, N, result)
        return result
