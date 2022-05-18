"""
simulation runners for partitioned simulations.
"""

import coupling_schemes as cs
import numpy as np
import timestepping as ts
from oscillator import PartitionedOscillator


def return_simulation_runner(coupling_scheme_str: str):
    if coupling_scheme_str == "css":
        run_simulation = cs.run_css_simulation
    elif coupling_scheme_str == "cps":
        run_simulation = cs.run_cps_simulation
    elif coupling_scheme_str == "strang":
        run_simulation = cs.run_strang_simulation
    elif coupling_scheme_str == "implicit-cps":
        run_simulation = cs.run_implicit_cps_simulation
    else:
        raise NotImplementedError(
            f"Coupling scheme {coupling_scheme_str} not implemented!"
        )
    return run_simulation


def return_system_partitions(
    t_end: float,
    N: int,
    result_values: int,
    coupling_scheme_str: str,
    **kwargs,
):
    if coupling_scheme_str == "strang":
        left_system = PartitionedOscillator(True, t_end, 2 * N, result_values, **kwargs)
        right_system = PartitionedOscillator(False, t_end, N, result_values, **kwargs)
    else:
        left_system = PartitionedOscillator(True, t_end, N, result_values, **kwargs)
        right_system = PartitionedOscillator(False, t_end, N, result_values, **kwargs)
    return left_system, right_system


def partitioned_newmark_beta(
    t_end: float,
    N: int,
    coupling_scheme_str: str = "",
    **kwargs,
):
    run_simulation = return_simulation_runner(coupling_scheme_str)
    left_system, right_system = return_system_partitions(
        t_end, N, 3, coupling_scheme_str, **kwargs
    )

    gamma = 0.5
    beta = 0.25
    # create solvers
    solver_left = ts.NewmarkBeta(
        left_system.A_second_order,
        left_system.M,
        left_system.K,
        beta,
        gamma,
        left_system.second_order_force,
    )
    solver_right = ts.NewmarkBeta(
        right_system.A_second_order,
        right_system.M,
        right_system.K,
        beta,
        gamma,
        right_system.second_order_force,
    )
    # run simulation
    left_result, right_result = run_simulation(
        left_system, solver_left, right_system, solver_right, t_end, N, **kwargs
    )
    full_result = np.array(
        [left_result[0], right_result[0], left_result[1], right_result[1]]
    )
    return full_result


def partitioned_generalized_alpha(
    t_end: float,
    N: int,
    coupling_scheme_str: str = "",
    **kwargs,
):
    run_simulation = return_simulation_runner(coupling_scheme_str)
    left_system, right_system = return_system_partitions(
        t_end, N, 3, coupling_scheme_str, **kwargs
    )

    alpha_m = 0.2
    alpha_f = 0.5
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (gamma + 0.5) ** 2
    # create solvers
    solver_left = ts.GeneralizedAlpha(
        left_system.A_second_order,
        left_system.M,
        left_system.K,
        beta,
        gamma,
        alpha_f,
        alpha_m,
        left_system.second_order_force,
    )
    solver_right = ts.GeneralizedAlpha(
        right_system.A_second_order,
        right_system.M,
        right_system.K,
        beta,
        gamma,
        alpha_f,
        alpha_m,
        right_system.second_order_force,
    )
    # run simulation
    left_result, right_result = run_simulation(
        left_system, solver_left, right_system, solver_right, t_end, N, **kwargs
    )
    full_result = np.array(
        [left_result[0], right_result[0], left_result[1], right_result[1]]
    )
    return full_result


def partitioned_erk(
    t_end: float,
    N: int,
    order: int,
    coupling_scheme_str: str = "",
    **kwargs,
):
    run_simulation = return_simulation_runner(coupling_scheme_str)
    left_system, right_system = return_system_partitions(
        t_end, N, 2, coupling_scheme_str, **kwargs
    )

    # create solvers
    solver_left = ts.ERK(
        left_system.A_first_order, left_system.first_order_force, order
    )
    solver_right = ts.ERK(
        right_system.A_first_order, right_system.first_order_force, order
    )
    left_result, right_result = run_simulation(
        left_system, solver_left, right_system, solver_right, t_end, N, **kwargs
    )
    full_result = np.array(
        [left_result[0], right_result[0], left_result[1], right_result[1]]
    )
    return full_result


def partitioned_semi_implicit_euler(
    t_end: float,
    N: int,
    coupling_scheme_str: str = "",
    **kwargs,
):
    run_simulation = return_simulation_runner(coupling_scheme_str)
    left_system, right_system = return_system_partitions(
        t_end, N, 2, coupling_scheme_str, **kwargs
    )

    # create solvers
    solver_left = ts.SemiImplicitEuler(
        left_system.A_first_order, left_system.second_order_force
    )
    solver_right = ts.SemiImplicitEuler(
        right_system.A_first_order, right_system.second_order_force
    )
    left_result, right_result = run_simulation(
        left_system, solver_left, right_system, solver_right, t_end, N, **kwargs
    )
    full_result = np.array(
        [left_result[0], right_result[0], left_result[1], right_result[1]]
    )
    return full_result


def partitioned_implicit_midpoint(
    t_end: float,
    N: int,
    coupling_scheme_str: str = "",
    **kwargs,
):
    run_simulation = return_simulation_runner(coupling_scheme_str)
    left_system, right_system = return_system_partitions(
        t_end, N, 2, coupling_scheme_str, **kwargs
    )

    # create solvers
    solver_left = ts.ImplicitMidpoint(
        left_system.A_first_order, left_system.first_order_force
    )
    solver_right = ts.ImplicitMidpoint(
        right_system.A_first_order, right_system.first_order_force
    )
    left_result, right_result = run_simulation(
        left_system, solver_left, right_system, solver_right, t_end, N, **kwargs
    )
    full_result = np.array(
        [left_result[0], right_result[0], left_result[1], right_result[1]]
    )
    return full_result
