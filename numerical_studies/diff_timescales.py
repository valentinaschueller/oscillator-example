from pathlib import Path

import numpy as np
from monolithic_system import MonolithicSystem
from system_partition import SystemPartition
from utility import plot_displacements, plot_energy, plot_velocities

m1 = 1
m2 = 1
k1 = 20
k2 = 0.1
k12 = 1
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
    w1 = 1.02463408140723
    w2 = 4.58804152108705
    result = np.array(
        [
            0.0262527968225597 * np.cos(w1 * t) + 0.473747203177441 * np.cos(w2 * t),
            0.523746578189158 * np.cos(w1 * t) - 0.0237465781891588 * np.cos(w2 * t),
            -0.0268995103566541 * np.sin(w1 * t) - 2.17357183867696 * np.sin(w2 * t),
            -0.536648594033029 * np.sin(w1 * t) + 0.108950286715601 * np.sin(w2 * t),
        ]
    )
    return result
