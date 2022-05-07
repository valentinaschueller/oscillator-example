from pathlib import Path

import numpy as np
from monolithic_system import MonolithicSystem
from system_partition import SystemPartition
from utility import plot_displacements, plot_energy, plot_velocities

m1 = 1
m2 = 1
k1 = 4*np.pi**2
k2 = 4*np.pi**2
k12 = 16*(np.pi**2)
M = np.array([[m1, 0], [0, m2]], dtype=float)
K = np.array([[(k1 + k12), -k12], [-k12, (k2 + k12)]], dtype=float)


def compute_energy(u1, u2, v1, v2):
    u_data = np.array([u1, u2])
    v_data = np.array([v1, v2])
    kinetic_energy = 0.5 * np.array([np.dot(v.T, np.dot(M, v)) for v in v_data.T])
    spring_energy = 0.5 * np.array([np.dot(u.T, np.dot(K, u)) for u in u_data.T])
    return kinetic_energy + spring_energy


def create_solution_plots(t, sol, dir_name: str = "plots"):
    plotdir_path = Path(dir_name)
    plotdir_path.mkdir(exist_ok=True)
    energy = compute_energy(sol[0, :], sol[1, :], sol[2, :], sol[3, :])
    plot_displacements(t, sol, plotdir_path)
    plot_velocities(t, sol, plotdir_path)
    plot_energy(t, energy, plotdir_path)


def analytical_solution(t_end: float, N: int):
    t = np.linspace(0, t_end, N + 1)
    result = np.array(
        [
            0.5 * (np.cos(2 * np.pi * t) + np.cos(6 * np.pi * t)),
            0.5 * (np.cos(2 * np.pi * t) - np.cos(6 * np.pi * t)),
            0.5 * (-2 * np.pi * np.sin(2 * np.pi * t) - 6 * np.pi * np.sin(6 * np.pi * t)),
            0.5 * (-2 * np.pi * np.sin(2 * np.pi * t) + 6 * np.pi * np.sin(6 * np.pi * t)),
        ]
    )
    return result


class Timescales(MonolithicSystem):
    def __init__(self):
        super().__init__(k1, k2, k12, m1, m2)

    def analytical_solution(self, t_end: float, N: int):
        result = analytical_solution(t_end, N)
        return result

    def _initial_conditions(self):
        u0 = np.array([1.0, 0.0])
        v0 = np.array([0.0, 0.0])
        a0 = np.dot(self.A_second_order, u0)
        return np.concatenate([u0, v0, a0])


class TimescalesPart(SystemPartition):
    def __init__(
        self,
        is_left_system: bool,
        t_end: float = 0.0,
        N: int = 0,
        result_values: int = 0,
        **kwargs
    ):
        super().__init__(
            is_left_system, t_end, N, result_values, k1, k2, k12, m1, m2, **kwargs
        )

    def _initial_conditions(self):
        if self.is_left_system:
            u0 = np.array([1.0])
        else:
            u0 = np.array([0.0])
        v0 = np.array([0.0])
        a0 = np.dot(self.A_second_order, u0) + self.k12 * self._initial_other_u()
        return np.concatenate([u0, v0, a0])

    def _initial_other_u(self):
        if not self.is_left_system:
            return 1.0
        else:
            return 0.0
