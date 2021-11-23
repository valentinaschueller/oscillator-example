import numpy as np
import matplotlib.pyplot as plt
from utility import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


from monolithic_system import MonolithicSystem
from timestepping import ERK, GeneralizedAlpha, NewmarkBeta

# k1 = k2 = m1 = m2 = 1

def run_simulation(t_stop: int, N: float, solver_str: str = "newmark"):
    ode_system = MonolithicSystem()
    
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
    else:
        solver = ERK(ode_system.A_first_order, force_function=ode_system.first_order_force , order=4)

    #t = np.linspace(0., t_stop, N+1)
    
    analytical_solution = ode_system.analytical_solution(t_stop, N)

    numerical_solution = ode_system.numerical_solution(t_stop, N, solver)

    return analytical_solution, numerical_solution

def compute_newmark_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="newmark")
    return true_sol - num_sol[0:4,:]

def compute_alpha_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="alpha")
    return true_sol - num_sol[0:4,:]

def compute_erk_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="erk")
    return true_sol - num_sol[0:4,:]

if __name__ == '__main__':
    # N = 100
    # t_stop = 20
    # true_sol, num_sol = run_simulation(t_stop, N, "erk")
    # t = np.linspace(0., t_stop, N+1)
    # create_solution_plots(t, num_sol, "erk_plots")
    # create_solution_plots(t, true_sol)
    
    t_stop = 20
    N_list = np.array([500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])
    l2_errors_newmark = np.array([l2_norm(compute_newmark_error(t_stop, N)) for N in N_list])
    l2_errors_alpha = np.array([l2_norm(compute_alpha_error(t_stop, N)) for N in N_list])
    l2_errors_erk = np.array([l2_norm(compute_erk_error(t_stop, N)) for N in N_list])

    title = "Monolithic System: Convergence Plot"
    subtitle = r"$\alpha_m = 0.2, \alpha_f = 0.5$"
    xlabel = "dt"
    ylabel = r"$\vert\vert e \vert\vert_2$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    ax.plot(dt_list, l2_errors_newmark, linestyle="none", marker=".", color="C9", label=r"Newmark $\beta$")
    ax.plot(dt_list, l2_errors_alpha, linestyle="none", marker="x", color="C6", label=r"Generalized $\alpha$")
    ax.plot(dt_list, l2_errors_erk, linestyle="none", marker="1", color="C8", label=r"ERK4")
    ax.legend()
    plt.savefig("convergence_plot_all.png", dpi=300, bbox_inches='tight')