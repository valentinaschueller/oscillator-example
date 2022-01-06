import numpy as np
import matplotlib.pyplot as plt

from utility import *
from monolithic_system import SameTimescales
from timestepping import ERK, GeneralizedAlpha, NewmarkBeta

# k1 = k2 = m1 = m2 = 1

def run_simulation(t_stop: int, N: float, solver_str: str = "newmark"):
    ode_system = SameTimescales()
    
    if solver_str == "newmark":
        newmark_gamma = 0.5
        newmark_beta = 0.25
        solver = NewmarkBeta(
            ode_system.A_second_order, ode_system.M, ode_system.K, 
            newmark_beta, newmark_gamma, 
            ode_system.second_order_force)
    elif solver_str == "alpha":
        alpha_m = 0.2
        alpha_f = 0.5
        gamma = 0.5 - alpha_m + alpha_f
        beta = 0.25 * (gamma + 0.5)**2
        solver = GeneralizedAlpha(
            ode_system.A_second_order, ode_system.M, ode_system.K, 
            beta, gamma, alpha_f, alpha_m, 
            ode_system.second_order_force)
    elif solver_str == "erk4":
        solver = ERK(ode_system.A_first_order, force_function=ode_system.first_order_force , order=4)
    elif solver_str == "erk1":
        solver = ERK(ode_system.A_first_order, force_function=ode_system.first_order_force , order=1)
    else:
        raise NotImplementedError(f"Solver {solver_str} not implemented!")
    
    analytical_solution = ode_system.analytical_solution(t_stop, N)

    numerical_solution = ode_system.numerical_solution(t_stop, N, solver)

    return analytical_solution, numerical_solution

def compute_newmark_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="newmark")
    return true_sol - num_sol[0:4,:]

def compute_alpha_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="alpha")
    return true_sol - num_sol[0:4,:]

def compute_erk_error(t_stop, N, order:int):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str=f"erk{order}")
    return true_sol - num_sol[0:4,:]

def beautify_plot(ax):
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    return ax

if __name__ == '__main__':
    t_stop = 20
    N_list = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])
    max_errors_newmark = np.array([max_norm(compute_newmark_error(t_stop, N)) for N in N_list])
    max_errors_alpha = np.array([max_norm(compute_alpha_error(t_stop, N)) for N in N_list])
    max_errors_erk4 = np.array([max_norm(compute_erk_error(t_stop, N, 4)) for N in N_list])
    max_errors_erk1 = np.array([max_norm(compute_erk_error(t_stop, N, 1)) for N in N_list])

    title = ""
    subtitle = "Monolithic System"
    xlabel = r"$\Delta t$"
    ylabel = r"$\left\| e \right\|_\infty$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    ax.plot(dt_list, max_errors_erk1, linestyle="none", marker="3", color="maroon", label=r"ERK1")
    ax.plot(dt_list, max_errors_newmark, linestyle="none", marker=".", color="darkcyan", label=r"Newmark-$\beta$")
    ax.plot(dt_list, max_errors_alpha, linestyle="none", marker="x", color="darkorchid", label=r"generalized-$\alpha$")
    ax.plot(dt_list, max_errors_erk4, linestyle="none", marker="1", color="olive", label=r"ERK4")
    ax.legend(ncol=2, loc='lower right')
    ax = beautify_plot(ax)
    plt.savefig("convergence_same_timescales_monolithic.pdf", dpi=300, bbox_inches='tight')