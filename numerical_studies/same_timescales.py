import numpy as np

from monolithic_system import MonolithicSystem
from system_partition import SystemPartition

m1 = 1
m2 = 1
k1 = 1
k2 = 1
k12 = 4
M = np.array([[m1, 0], [0, m2]], dtype=float)
K = np.array([[(k1 + k12), -k12], [-k12, (k2 + k12)]], dtype=float)


def compute_energy(u1, u2, v1, v2):
    u_data = np.array([u1, u2])
    v_data = np.array([v1, v2])
    kinetic_energy = 0.5 * np.array([np.dot(v.T, np.dot(M, v)) for v in v_data.T])
    spring_energy = 0.5 * np.array([np.dot(u.T, np.dot(K, u)) for u in u_data.T])
    return kinetic_energy + spring_energy


def analytical_solution(t_end: float, N: int):
    t = np.linspace(0, t_end, N + 1)
    result = np.array(
        [
            0.5 * (np.cos(t) + np.cos(3 * t)),
            0.5 * (np.cos(t) - np.cos(3 * t)),
            0.5 * (-np.sin(t) - 3 * np.sin(3 * t)),
            0.5 * (-np.sin(t) + 3 * np.sin(3 * t)),
        ]
    )
    return result


class SameTimescales(MonolithicSystem):
    def __init__(self):
        super().__init__(1, 1, 4, 1, 1)

    def analytical_solution(self, t_end: float, N: int):
        result = analytical_solution(t_end, N)
        return result

    def _initial_conditions(self):
        u0 = np.array([1.0, 0.0])
        v0 = np.array([0.0, 0.0])
        a0 = np.dot(self.A_second_order, u0)
        return np.concatenate([u0, v0, a0])


class SameTimescalesPart(SystemPartition):
    def __init__(
        self,
        left_system: bool,
        t_end: float = 0.0,
        N: int = 0,
        result_values: int = 0,
        **kwargs
    ):
        super().__init__(left_system, t_end, N, result_values, 1, 1, 4, 1, 1, **kwargs)

    def _initial_conditions(self):
        if self.left_system_bool:
            u0 = np.array([1.0])
        else:
            u0 = np.array([0.0])
        v0 = np.array([0.0])
        a0 = np.dot(self.A_second_order, u0) + self.k12 * self._initial_other_u()
        return np.concatenate([u0, v0, a0])

    def _initial_other_u(self):
        if not self.left_system_bool:
            return 1.0
        else:
            return 0.0
