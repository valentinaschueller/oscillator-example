import matplotlib.pyplot as plt
import numpy as np
from run_partitioned_simulation import *
from same_timescales import SameTimescalesPart, analytical_solution
from utility import max_norm, plot_error_ref, prepare_plot


def compute_newmark_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_newmark_beta(
        t_stop, N, SameTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def compute_alpha_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_generalized_alpha(
        t_stop, N, SameTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def compute_erk4_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(
        t_stop, N, 4, SameTimescalesPart, coupling_scheme_str, **kwargs
    )
    return true_sol - num_sol


def compute_erk1_error(t_stop, N, coupling_scheme_str: str = "", **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(
        t_stop, N, 1, SameTimescalesPart, coupling_scheme_str, **kwargs
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
    # result = partitioned_erk(t_stop, N, 4, "implicit-cps")

    t_stop = 20
    N_list = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])
    errors_newmark_wi = np.array(
        [
            max_norm(
                compute_newmark_error(t_stop, N, "implicit-cps", interpolation_order=1)
            )
            for N in N_list
        ]
    )
    errors_alpha_wi = np.array(
        [
            max_norm(
                compute_alpha_error(t_stop, N, "implicit-cps", interpolation_order=1)
            )
            for N in N_list
        ]
    )
    errors_erk4_wi = np.array(
        [
            max_norm(
                compute_erk4_error(t_stop, N, "implicit-cps", interpolation_order=1)
            )
            for N in N_list
        ]
    )
    errors_erk1_wi = np.array(
        [
            max_norm(
                compute_erk1_error(t_stop, N, "implicit-cps", interpolation_order=1)
            )
            for N in N_list
        ]
    )

    title = ""
    subtitle = "Waveform Iterations (Linear)"
    xlabel = r"$\Delta t$"
    ylabel = r"$\left\| e \right\|_\infty$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    ax.plot(
        dt_list,
        errors_erk1_wi,
        linestyle="none",
        marker="3",
        color="maroon",
        label=r"ERK1",
    )
    ax.plot(
        dt_list,
        errors_newmark_wi,
        linestyle="none",
        marker=".",
        color="darkcyan",
        label=r"Newmark-$\beta$",
    )
    ax.plot(
        dt_list,
        errors_alpha_wi,
        linestyle="none",
        marker="x",
        color="darkorchid",
        label=r"generalized-$\alpha$",
    )
    ax.plot(
        dt_list,
        errors_erk4_wi,
        linestyle="none",
        marker="1",
        color="olive",
        label=r"ERK4",
    )

    ax.legend(ncol=2, loc="lower right")
    ax = beautify_plot(ax)
    plt.savefig(
        "convergence_same_timescales_partitioned_waveform.pdf",
        dpi=300,
        bbox_inches="tight",
    )