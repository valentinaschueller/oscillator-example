import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core import numeric
from plotting import prepare_plot, plot_error_ref

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


def compute_energy(solution: np.ndarray):
    u1 = solution[0,:]
    u2 = solution[1,:]
    v1 = solution[2,:]
    v2 = solution[3,:]
    return v1**2 + v2**2 + u1**2 + u2**2 + (u2-u1)**2


def plot_displacements(t, sol, path="."):
    _, ax = prepare_plot("Displacements", "", "t [s]", "u [m]")
    ax.plot(t, sol[0,:], label="$u_1$")
    ax.plot(t, sol[1,:], label="$u_2$")
    ax.legend()
    plt.savefig(f"{path}/displacements.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_velocities(t, sol, path="."):
    _, ax = prepare_plot("Velocities", "", "t [s]", "v [m/s]")
    ax.plot(t, sol[2,:], label="$v_1$")
    ax.plot(t, sol[3,:], label="$v_2$")
    ax.legend()
    plt.savefig(f"{path}/velocities.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_energy(t, energy, path="."):
    _, ax = prepare_plot("Energy", "", "t [s]", "Energy")
    ax.plot(t, energy, label='energy')
    ax.legend()
    plt.savefig(f"{path}/energy.png", dpi=300, bbox_inches="tight")
    plt.close()

def l1_norm(vec):
    return np.sum(np.abs(vec))/vec.shape[1]

def l2_norm(vec):
    return np.sum(vec**2)/vec.shape[1]

def max_norm(vec):
    return np.max(np.abs(vec))

def compute_newmark_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="newmark")
    return true_sol - num_sol[0:4,:]

def compute_alpha_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="alpha")
    return true_sol - num_sol[0:4,:]

def compute_erk_error(t_stop, N):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str="erk")
    return true_sol - num_sol[0:4,:]

def create_solution_plots(t, sol, dir_name: str ="plots"):
    plotdir_path = f"./{dir_name}"
    try:
        os.mkdir(plotdir_path)
    except FileExistsError:
        pass
    energy = compute_energy(sol)
    plot_displacements(t, sol, plotdir_path)
    plot_velocities(t, sol, plotdir_path)
    plot_energy(t, energy, plotdir_path)

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
    l2_errors_newmark = np.array([l1_norm(compute_newmark_error(t_stop, N)) for N in N_list])
    l2_errors_alpha = np.array([l1_norm(compute_alpha_error(t_stop, N)) for N in N_list])
    l2_errors_erk = np.array([l1_norm(compute_erk_error(t_stop, N)) for N in N_list])

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