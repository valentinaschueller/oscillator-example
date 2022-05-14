"""
analytical solution and ODE system classes for the monolithic and partitioned oscillator.

the analytical solution has to be changed manually but the mass/stiffness parameters can be
changed at the top of the module.
"""

from pathlib import Path

import numpy as np

from timestepping import TimesteppingMethod
from utility import interpolate_linear, plot_displacements, plot_energy, plot_velocities

# mass and stiffness parameters
m1 = 1
m2 = 1
k1 = 4 * np.pi**2
k2 = 4 * np.pi**2
k12 = 16 * (np.pi**2)
M = np.array([[m1, 0], [0, m2]], dtype=float)
K = np.array([[(k1 + k12), -k12], [-k12, (k2 + k12)]], dtype=float)

# initial conditions
u1_0 = 1.0
u2_0 = 0.0
v1_0 = 0.0
v2_0 = 0.0


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
            0.5
            * (-2 * np.pi * np.sin(2 * np.pi * t) - 6 * np.pi * np.sin(6 * np.pi * t)),
            0.5
            * (-2 * np.pi * np.sin(2 * np.pi * t) + 6 * np.pi * np.sin(6 * np.pi * t)),
        ]
    )
    return result


class MonolithicOscillator:
    def __init__(self):
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

    def _initial_conditions(self):
        u0 = np.array([u1_0, u2_0])
        v0 = np.array([v1_0, v2_0])
        a0 = np.dot(self.A_second_order, u0)
        return np.concatenate([u0, v0, a0])

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

    def analytical_solution(self, t_end: float, N: int):
        result = analytical_solution(t_end, N)
        return result


class PartitionedOscillator:
    def __init__(
        self,
        is_left_system: bool,
        t_end: float = 0.0,
        N: int = 0,
        result_values: int = 0,
        **kwargs
    ):
        self.is_left_system = is_left_system
        self.k12 = k12
        self.interpolation_order = kwargs.get("interpolation_order", 0)

        if self.is_left_system:
            # for the formulation: Mu'' + Ku = 0:
            self.M = m1 * np.eye(1, 1, dtype=float)
            self.K = (k1 + k12) * np.eye(1, 1, dtype=float)
            # for the formulation: y' = A_first_order * y:
            self.A_first_order = np.array([[0.0, 1.0], [-(k1 + k12) / m1, 0.0]])
            # for the formulation: u'' = A_second_order * u:
            self.A_second_order = -(k1 + k12) * np.eye(1, 1, dtype=float)
        else:
            # for the formulation: Mu'' + Ku = 0:
            self.M = m2 * np.eye(1, 1, dtype=float)
            self.K = (k2 + k12) * np.eye(1, 1, dtype=float)
            # for the formulation: y' = A_first_order * y:
            self.A_first_order = np.array([[0.0, 1.0], [-(k2 + k12) / m2, 0.0]])
            # for the formulation: u'' = A_second_order * u:
            self.A_second_order = -(k2 + k12) * np.eye(1, 1, dtype=float)

        self.result = np.full((result_values, N + 1), np.inf)
        self.t_values = np.linspace(0, t_end, N + 1)
        self.dt = t_end / N
        self.other_u = np.full(N + 1, np.inf)
        self.other_u[0] = self._initial_other_u()
        try:
            self.result[:, 0] = self._initial_conditions()
        except ValueError:
            self.result[:, 0] = self._initial_conditions()[:2]

    def other_u_at(self, t, t_lower):
        # get id of "left" u-value in current interval
        idx = np.where(np.abs(self.t_values - t_lower) < 1e-6)[0][0]
        if self.interpolation_order == 0:
            if self.other_u[idx + 1] == np.inf:
                self.other_u[idx + 1] = self.other_u[idx]
            del t
            return self.other_u[idx + 1]
        elif self.interpolation_order == 1:
            previous_other_u = self.other_u[idx]
            if self.other_u[idx + 1] == np.inf:
                self.other_u[
                    idx + 1
                ] = previous_other_u  # constant extrapolation in first iteration
            next_other_u = self.other_u[idx + 1]
            percentage = (t - t_lower) / self.dt
            interpolated_u = interpolate_linear(
                previous_other_u, next_other_u, percentage
            )
            return interpolated_u
        else:
            raise NotImplementedError

    def second_order_force(self, t, t_lower):
        if t == 0:
            return self.k12 * self._initial_other_u()
        else:
            return self.k12 * self.other_u_at(t, t_lower)

    def first_order_force(self, t, t_lower):
        if t == 0:
            return np.array([0.0, self.k12 * self._initial_other_u()], dtype=object)
        else:
            return np.array([0.0, self.k12 * self.other_u_at(t, t_lower)], dtype=object)

    def _initial_conditions(self):
        if self.is_left_system:
            u0 = np.array([u1_0])
            v0 = np.array([v1_0])
        else:
            u0 = np.array([u2_0])
            v0 = np.array([v2_0])
        a0 = np.dot(self.A_second_order, u0) + self.k12 * self._initial_other_u()
        return np.concatenate([u0, v0, a0])

    def _initial_other_u(self):
        if not self.is_left_system:
            return 1.0
        else:
            return 0.0
