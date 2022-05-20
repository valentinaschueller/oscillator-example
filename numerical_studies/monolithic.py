import numpy as np
import pandas as pd
from oscillator import MonolithicOscillator
from timestepping import (ERK, GeneralizedAlpha, ImplicitMidpoint, NewmarkBeta,
                          SemiImplicitEuler)
from utility import comment_meta_information, max_norm


def run_simulation(t_stop: int, N: float, solver_str: str = "newmark"):
    ode_system = MonolithicOscillator()

    if solver_str == "newmark":
        newmark_gamma = 0.5
        newmark_beta = 0.25
        solver = NewmarkBeta(
            ode_system.A_second_order,
            ode_system.M,
            ode_system.K,
            newmark_beta,
            newmark_gamma,
            ode_system.second_order_force,
        )
    elif solver_str == "alpha":
        alpha_m = 0.2
        alpha_f = 0.5
        gamma = 0.5 - alpha_m + alpha_f
        beta = 0.25 * (gamma + 0.5) ** 2
        solver = GeneralizedAlpha(
            ode_system.A_second_order,
            ode_system.M,
            ode_system.K,
            beta,
            gamma,
            alpha_f,
            alpha_m,
            ode_system.second_order_force,
        )
    elif solver_str == "erk1":
        solver = ERK(
            ode_system.A_first_order,
            ode_system.first_order_force,
            order=1,
        )
    elif solver_str == "heun":
        solver = ERK(
            ode_system.A_first_order,
            ode_system.first_order_force,
            order=2,
        )
    elif solver_str == "erk4":
        solver = ERK(
            ode_system.A_first_order,
            ode_system.first_order_force,
            order=4,
        )
    elif solver_str == "mid":
        solver = ImplicitMidpoint(
            ode_system.A_first_order,
            ode_system.first_order_force,
        )
    elif solver_str == "sie":
        solver = SemiImplicitEuler(
            ode_system.A_first_order,
            ode_system.second_order_force,
        )
    else:
        raise NotImplementedError(f"Solver {solver_str} not implemented!")

    analytical_solution = ode_system.analytical_solution(t_stop, N)

    numerical_solution = ode_system.numerical_solution(t_stop, N, solver)

    return analytical_solution, numerical_solution


def compute_simulation_error(t_stop, N, solver_str):
    true_sol, num_sol = run_simulation(t_stop, N, solver_str)
    return true_sol - num_sol[0:4, :]


if __name__ == "__main__":
    """
    Runs monolithic experiment for oscillator example with different time stepping schemes.
    Performs convergence study and outputs error w.r.t analytical solution.
    """
    # t_stop = 20
    # N_list = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    t_stop = 1
    N_list = np.array([25, 50, 100, 200, 400, 800, 1600, 3200, 6400])
    dt_list = np.array([t_stop / N for N in N_list])

    # prepare dataframe for saving
    errors_df = pd.DataFrame(index=dt_list)
    errors_df.index.name = "dt"

    method_names = ["newmark", "alpha", "erk4", "sie", "mid"]
    for method_name in method_names:
        errors_df["error"] = np.array(
            [max_norm(compute_simulation_error(t_stop, N, method_name)) for N in N_list]
        )
        filename = f"results/monolithic_{method_name}.csv"
        errors_df.to_csv(filename)
        comment_meta_information(method_name, __file__, filename)
