import numpy as np
import time

from utility import *
from system_partition import SystemPartition
from coupling_schemes import *
from timestepping import ERK, GeneralizedAlpha, NewmarkBeta

import matplotlib.pyplot as plt

def analytical_solution(t_end: float, N: int):
    t = np.linspace(0, t_end, N+1)
    result = np.array([
        0.5 * (np.cos(t) + np.cos(np.sqrt(3)*t)),
        0.5 * (np.cos(t) - np.cos(np.sqrt(3)*t)),
        0.5 * (- np.sin(t) - np.sqrt(3) * np.sin(np.sqrt(3)*t)),
        0.5 * (- np.sin(t) + np.sqrt(3) * np.sin(np.sqrt(3)*t)),
    ])
    return result

def partitioned_newmark_beta(t_end: float, N: int, coupling_scheme_str: str = ""):
    if coupling_scheme_str == "css":
        run_simulation = run_css_simulation
        left_system = SystemPartition(left_system=True, result_shape=(3, N+1))
        right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "cps":
        run_simulation = run_cps_simulation
        left_system = SystemPartition(left_system=True, result_shape=(3, N+1))
        right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "strang":
        run_simulation = run_strang_simulation
        left_system = SystemPartition(left_system=True, result_shape=(3, 2*N+1))
        right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    else:
        raise NotImplementedError(f"Coupling scheme {coupling_scheme_str} not implemented!")
    gamma = 0.5
    beta = 0.25
    # create solvers
    solver_left = NewmarkBeta(
        left_system.A_second_order, left_system.M, left_system.K,
        beta, gamma, left_system.second_order_force
    )
    solver_right = NewmarkBeta(
        right_system.A_second_order, right_system.M, right_system.K,
        beta, gamma, right_system.second_order_force
    )
    # run simulation
    left_result, right_result = run_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def partitioned_generalized_alpha(t_end: float, N: int, coupling_scheme_str: str = ""):
    if coupling_scheme_str == "css":
        run_simulation = run_css_simulation
        left_system = SystemPartition(left_system=True, result_shape=(3, N+1))
        right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "cps":
        run_simulation = run_cps_simulation
        left_system = SystemPartition(left_system=True, result_shape=(3, N+1))
        right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "strang":
        run_simulation = run_strang_simulation
        left_system = SystemPartition(left_system=True, result_shape=(3, 2*N+1))
        right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    else:
        raise NotImplementedError(f"Coupling scheme {coupling_scheme_str} not implemented!")
    alpha_m = 0.5
    alpha_f = 0.5
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (gamma + 0.5)**2
    # create solvers
    solver_left = GeneralizedAlpha(
        left_system.A_second_order, left_system.M, left_system.K,
        beta, gamma, alpha_f, alpha_m, left_system.second_order_force
    )
    solver_right = GeneralizedAlpha(
        right_system.A_second_order, right_system.M, right_system.K,
        beta, gamma, alpha_f, alpha_m, right_system.second_order_force
    )
    # run simulation
    left_result, right_result = run_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def partitioned_erk(t_end: float, N: int, order: int = 1, coupling_scheme_str: str = ""):
    if coupling_scheme_str == "css":
        run_simulation = run_css_simulation
        left_system = SystemPartition(left_system=True, result_shape=(2, N+1))
        right_system = SystemPartition(left_system=False, result_shape=(2, N+1))
    elif coupling_scheme_str == "cps":
        run_simulation = run_cps_simulation
        left_system = SystemPartition(left_system=True, result_shape=(2, N+1))
        right_system = SystemPartition(left_system=False, result_shape=(2, N+1))
    elif coupling_scheme_str == "strang":
        run_simulation = run_strang_simulation
        left_system = SystemPartition(left_system=True, result_shape=(2, 2*N+1))
        right_system = SystemPartition(left_system=False, result_shape=(2, N+1))
    else:
        raise NotImplementedError(f"Coupling scheme {coupling_scheme_str} not implemented!")
    # create solvers
    solver_left = ERK(left_system.A_first_order, left_system.first_order_force, order)
    solver_right = ERK(right_system.A_first_order, right_system.first_order_force, order)
    left_result, right_result = run_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def compute_newmark_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_newmark_beta(t_stop, N, coupling_scheme_str)
    return true_sol - num_sol

def compute_alpha_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_generalized_alpha(t_stop, N, coupling_scheme_str)
    return true_sol - num_sol

def compute_erk_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(t_stop, N, 4, coupling_scheme_str)
    return true_sol - num_sol

if __name__ == '__main__':
    # N = 100
    # t_stop = 20
    # true_sol = analytical_solution(t_stop, N)
    # num_sol_alpha = partitioned_generalized_alpha(t_stop, N, "cps")
    # num_sol_erk4 = partitioned_erk(t_stop, N, 4, "cps")
    # t = np.linspace(0., t_stop, N+1)
    # create_solution_plots(t, num_sol_alpha, f"partitioned_alpha_{N}")
    # create_solution_plots(t, num_sol_erk4, f"partitioned_erk_{N}")
    # create_solution_plots(t, true_sol, "analytic")

    t_stop = 20
    N_list = np.array([500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])
    errors_newmark = np.array([max_norm(compute_newmark_error(t_stop, N, "cps")) for N in N_list])
    errors_alpha = np.array([max_norm(compute_alpha_error(t_stop, N, "cps")) for N in N_list])
    errors_alpha_strang = np.array([max_norm(compute_alpha_error(t_stop, N, "strang")) for N in N_list])
    errors_erk = np.array([max_norm(compute_erk_error(t_stop, N, "cps")) for N in N_list])
    errors_erk_strang = np.array([max_norm(compute_erk_error(t_stop, N, "strang")) for N in N_list])

    title = "Partitioned System: Convergence Plot"
    subtitle = r"$\alpha_m = 0.3, \alpha_f = 0.5$"
    xlabel = "dt"
    ylabel = r"$\vert\vert e \vert\vert_\infty$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    ax.plot(dt_list, errors_newmark, linestyle="none", marker=".", color="C9", label=r"Newmark $\beta$-CPS")
    ax.plot(dt_list, errors_alpha, linestyle="none", marker="x", color="C6", label=r"Generalized $\alpha$-CPS")
    ax.plot(dt_list, errors_erk, linestyle="none", marker="1", color="C8", label=r"ERK4-CPS")
    ax.plot(dt_list, errors_erk_strang, linestyle="none", marker="1", color="olive", label=r"ERK4-Strang")
    ax.plot(dt_list, errors_alpha_strang, linestyle="none", marker="x", color="darkcyan", label=r"Alpha-Strang")
    ax.legend()
    plt.savefig("convergence_partitioned_all.png", dpi=300, bbox_inches='tight')