import numpy as np

from utility import plot_error_ref, prepare_plot, max_norm
from same_timescales import SameTimescalesPart, analytical_solution
from run_partitioned_simulation import *

import matplotlib.pyplot as plt


def compute_semi_implicit_euler_error(t_stop, N, coupling_scheme_str: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_semi_implicit_euler(t_stop, N, SameTimescalesPart, coupling_scheme_str, **kwargs)
    return true_sol - num_sol


def beautify_plot(ax):
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    return ax


def plot_error_ref(ax, dt):
    """create an error log-log plot in the current axis"""
    o1 = dt
    o2 = dt * dt
    o3 = dt**3
    o4 = dt**4
    ax.loglog(dt, o1, "k-", label="$\mathcal{O}(\Delta t)$", lw=1)
    ax.loglog(dt, o2, "k--", label="$\mathcal{O}(\Delta t^2)$", lw=1)
    ax.loglog(dt, o3, "k:", label="$\mathcal{O}(\Delta t^3)$", lw=1)
    ax.loglog(dt, o4, "k-.", label="$\mathcal{O}(\Delta t^4)$", lw=1)
    return ax


if __name__ == "__main__":
    t_stop = 20
    N_list = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])
    # errors_newmark_cps = np.array([max_norm(compute_newmark_error(t_stop, N, "cps")) for N in N_list])
    # errors_mid_cps = np.array(
    #     [max_norm(compute_implicit_midpoint_error(t_stop, N, "cps")) for N in N_list]
    # )
    errors_sie_cps = np.array(
        [max_norm(compute_semi_implicit_euler_error(t_stop, N, "cps")) for N in N_list]
    )
    errors_sie_css = np.array(
        [max_norm(compute_semi_implicit_euler_error(t_stop, N, "css")) for N in N_list]
    )
    errors_sie_icps = np.array(
        [max_norm(compute_semi_implicit_euler_error(t_stop, N, "implicit-cps")) for N in N_list]
    )
    errors_sie_strang = np.array(
        [max_norm(compute_semi_implicit_euler_error(t_stop, N, "strang")) for N in N_list]
    )
    errors_sie_waveform = np.array(
        [max_norm(compute_semi_implicit_euler_error(t_stop, N, "implicit-cps", interpolation_order=1)) for N in N_list]
    )

    title = ""
    subtitle = "Symplectic Euler: Different coupling schemes"
    xlabel = r"$\Delta t$"
    ylabel = r"$\left\| e \right\|_\infty$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    # ax.plot(
    #     dt_list,
    #     errors_mid_cps,
    #     linestyle="none",
    #     marker="3",
    #     color="maroon",
    #     label=r"Implicit Midpoint (CPS)",
    # )
    ax.plot(
        dt_list,
        errors_sie_cps,
        linestyle="none",
        marker="x",
        color="tomato",
        label=r"CPS",
    )
    ax.plot(
        dt_list,
        errors_sie_css,
        linestyle="none",
        marker="2",
        color="C8",
        label=r"CSS",
    )
    ax.plot(
        dt_list,
        errors_sie_icps,
        linestyle="none",
        marker="4",
        color="maroon",
        label=r"Implicit CPS",
    )
    ax.plot(
        dt_list,
        errors_sie_waveform,
        linestyle="none",
        marker="3",
        color="green",
        label=r"Waveform iterations",
    )
    ax.plot(
        dt_list,
        errors_sie_strang,
        linestyle="none",
        marker=".",
        color="blue",
        label=r"Strang splitting",
    )

    ax.legend(ncol=2, loc="lower right")
    ax = beautify_plot(ax)
    plt.savefig(
        "symplectic_euler.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Phase plots

    t_end = 40
    N = 500
    t = np.linspace(0, t_end, N + 1)

    fig, axs = plt.subplots(1, 5, figsize=(35,10))
    fig.suptitle(rf'Phase plots semi-implicit Euler, N = {N}, $t_E$ = {t_end}', fontsize=14)

    nsol = partitioned_semi_implicit_euler(t_end, N, SameTimescalesPart, "cps")
    u1 = nsol[0]
    v1 = nsol[2]
    axs[0].plot(u1, v1)
    axs[0].set_title("CPS")

    nsol = partitioned_semi_implicit_euler(t_end, N, SameTimescalesPart, "css")
    u1 = nsol[0]
    v1 = nsol[2]
    axs[1].plot(u1, v1)
    axs[1].set_title("CSS")

    nsol = partitioned_semi_implicit_euler(t_end, N, SameTimescalesPart, "implicit-cps")
    u1 = nsol[0]
    v1 = nsol[2]
    axs[2].plot(u1, v1)
    axs[2].set_title("Implicit CPS")

    nsol = partitioned_semi_implicit_euler(t_end, N, SameTimescalesPart, "implicit-cps", interpolation_order=1)
    u1 = nsol[0]
    v1 = nsol[2]
    axs[3].plot(u1, v1)
    axs[3].set_title("Waveform iterations")

    nsol = partitioned_semi_implicit_euler(t_end, N, SameTimescalesPart, "strang")
    u1 = nsol[0]
    v1 = nsol[2]
    axs[4].plot(u1, v1)
    axs[4].set_title("Strang")

    for ax in axs:
        ax.set_xlabel(r"$u_1$")
        ax.set_ylabel(r"$v_1$")

    plt.savefig("phase_space_sie.png", dpi=300, bbox_inches="tight")
