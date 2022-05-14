"""
collection of compute_*_error() functions for the partitioned experiments
"""

import run_partitioned_simulation as rps
from oscillator import analytical_solution


def compute_newmark_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_newmark_beta(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol


def compute_alpha_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_generalized_alpha(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol


def compute_erk4_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_erk(t_stop, N, 4, coupling_scheme, **kwargs)
    return true_sol - num_sol


def compute_sie_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_semi_implicit_euler(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol


def compute_mid_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_implicit_midpoint(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol
