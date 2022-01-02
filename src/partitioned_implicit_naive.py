import numpy as np
import time

from utility import *
from run_partitioned_simulation import *
from coupling_schemes import *

import matplotlib.pyplot as plt


def compute_newmark_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_newmark_beta(t_stop, N, coupling_scheme_str)
    return true_sol - num_sol

def compute_alpha_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_generalized_alpha(t_stop, N, coupling_scheme_str)
    return true_sol - num_sol

def compute_erk4_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(t_stop, N, 4, coupling_scheme_str)
    return true_sol - num_sol

def compute_erk1_error(t_stop, N, coupling_scheme_str: str = ""):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(t_stop, N, 1, coupling_scheme_str)
    return true_sol - num_sol

def beautify_plot(ax):
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

if __name__ == '__main__':
    t_stop = 20
    N_list = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])
    errors_newmark_cps = np.array([max_norm(compute_newmark_error(t_stop, N, "implicit-cps")) for N in N_list])
    errors_alpha_cps = np.array([max_norm(compute_alpha_error(t_stop, N, "implicit-cps")) for N in N_list])
    # errors_newmark_css = np.array([max_norm(compute_newmark_error(t_stop, N, "implicit-css")) for N in N_list])
    errors_erk4_cps = np.array([max_norm(compute_erk4_error(t_stop, N, "implicit-cps")) for N in N_list])
    # errors_erk4_css = np.array([max_norm(compute_erk4_error(t_stop, N, "implicit-css")) for N in N_list])
    errors_erk1_cps = np.array([max_norm(compute_erk1_error(t_stop, N, "implicit-cps")) for N in N_list])
    # errors_erk1_css = np.array([max_norm(compute_erk1_error(t_stop, N, "implicit-css")) for N in N_list])

    title = "Partitioned System (same time scales): Convergence Study"
    subtitle = "Naive Implicit Coupling Schemes"
    xlabel = r"$\Delta t$"
    ylabel = r"$\left\| e \right\|_\infty$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    ax.plot(dt_list, errors_newmark_cps, linestyle="none", marker="3", color="maroon", label=r"Newmark-FP-CPS")
    ax.plot(dt_list, errors_alpha_cps, linestyle="none", marker="1", color="darkcyan", label=r"generalized-$\alpha$-FP-CPS")
    # ax.plot(dt_list, errors_newmark_css, linestyle="none", marker=".", color="darkcyan", label=r"Newmark-CSS")
    ax.plot(dt_list, errors_erk1_cps, linestyle="none", marker="+", color="green", label=r"ERK1-FP-CPS")
    # ax.plot(dt_list, errors_erk1_css, linestyle="none", marker="4", color="red", label=r"ERK1-CSS")
    ax.plot(dt_list, errors_erk4_cps, linestyle="none", marker="x", color="darkorchid", label=r"ERK4-FP-CPS")
    # ax.plot(dt_list, errors_erk4_css, linestyle="none", marker="1", color="olive", label=r"ERK4-CSS")

    ax.legend(ncol=2, loc='lower right')
    ax = beautify_plot(ax)
    plt.savefig("convergence_same_timescales_monolithic_implicit.pdf", dpi=300, bbox_inches='tight')