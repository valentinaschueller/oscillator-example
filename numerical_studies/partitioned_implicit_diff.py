import numpy as np

from utility import prepare_plot, plot_error_ref, max_norm
from run_partitioned_simulation import *
from diff_timescales import DiffTimescalesPart, analytical_solution

import matplotlib.pyplot as plt


def compute_newmark_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_newmark_beta(
        t_stop, N, DiffTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def compute_alpha_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_generalized_alpha(
        t_stop, N, DiffTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def compute_erk4_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(
        t_stop, N, 4, DiffTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def compute_erk1_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(
        t_stop, N, 1, DiffTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def beautify_plot(ax):
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    return ax


if __name__ == "__main__":
    # t_stop = 1
    # N = 2
    # erk_sol = partitioned_erk(t_stop, N, 4, "implicit-cps", sc=5, interpolation_order=1)
    # create_solution_plots(np.linspace(0,t_stop,N+1), erk_sol, "implicit-subcycling-erk4-1")
    # create_solution_plots(np.linspace(0,t_stop,N+1), analytical_solution(t_stop, N), "analytical")

    t_stop = 20
    N_list = np.array([250, 500, 1000, 2000, 4000])
    dt_list = np.array([t_stop / N for N in N_list])
    # errors_newmark_cps = np.array([max_norm(compute_newmark_error(t_stop, N, "implicit-cps", sc=4, interpolation_order=1)) for N in N_list])
    # errors_alpha_cps = np.array([max_norm(compute_alpha_error(t_stop, N, "implicit-cps", sc=4, interpolation_order=1)) for N in N_list])
    # errors_newmark_css = np.array([max_norm(compute_newmark_error(t_stop, N, "implicit-css")) for N in N_list])
    errors_erk4_cps = np.array(
        [
            max_norm(
                compute_erk4_error(
                    t_stop, N, "implicit-cps", sc=4, interpolation_order=1
                )
            )
            for N in N_list
        ]
    )
    errors_erk4_cps_io_0 = np.array(
        [
            max_norm(
                compute_erk4_error(
                    t_stop, N, "implicit-cps", sc=4, interpolation_order=0
                )
            )
            for N in N_list
        ]
    )
    # errors_erk4_css = np.array([max_norm(compute_erk4_error(t_stop, N, "implicit-css")) for N in N_list])
    # errors_erk1_cps = np.array([max_norm(compute_erk1_error(t_stop, N, "implicit-cps", sc=4, interpolation_order=1)) for N in N_list])
    # errors_erk1_css = np.array([max_norm(compute_erk1_error(t_stop, N, "implicit-css")) for N in N_list])

    title = ""
    subtitle = "Naive Implicit Coupling Schemes"
    xlabel = r"$\Delta t$"
    ylabel = r"$\left\| e \right\|_\infty$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    # ax.plot(dt_list, errors_newmark_cps, linestyle="none", marker="3", color="maroon", label=r"Newmark-FP-CPS")
    # ax.plot(dt_list, errors_alpha_cps, linestyle="none", marker="1", color="darkcyan", label=r"generalized-$\alpha$-FP-CPS")
    # ax.plot(dt_list, errors_newmark_css, linestyle="none", marker=".", color="darkcyan", label=r"Newmark-CSS")
    # ax.plot(dt_list, errors_erk1_cps, linestyle="none", marker="+", color="green", label=r"ERK1-FP-CPS")
    # ax.plot(dt_list, errors_erk1_css, linestyle="none", marker="4", color="red", label=r"ERK1-CSS")
    ax.plot(
        dt_list,
        errors_erk4_cps,
        linestyle="none",
        marker="x",
        color="darkorchid",
        label=r"ERK4-1",
    )
    ax.plot(
        dt_list,
        errors_erk4_cps_io_0,
        linestyle="none",
        marker="x",
        color="darkcyan",
        label=r"ERK4-0",
    )
    # ax.plot(dt_list, errors_erk4_css, linestyle="none", marker="1", color="olive", label=r"ERK4-CSS")

    ax.legend(ncol=2, loc="lower right")
    ax = beautify_plot(ax)
    plt.savefig(
        "convergence_diff_timescales_partitioned_implicit_waveform.png",
        dpi=300,
        bbox_inches="tight",
    )
