import numpy as np
import pandas as pd
import run_partitioned_simulation as rps
from timescales import TimescalesPart, analytical_solution
from utility import max_norm


def compute_newmark_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_newmark_beta(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return true_sol - num_sol


def compute_alpha_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_generalized_alpha(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return true_sol - num_sol


def compute_erk4_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_erk(
        t_stop, N, 4, TimescalesPart, coupling_scheme, **kwargs
    )
    return true_sol - num_sol


def compute_sie_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_semi_implicit_euler(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return true_sol - num_sol


def compute_mid_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol = rps.partitioned_implicit_midpoint(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return true_sol - num_sol


if __name__ == "__main__":
    t_stop = 20
    N_list = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    dt_list = np.array([t_stop / N for N in N_list])

    # prepare dataframe for saving
    errors_df = pd.DataFrame(index=dt_list)
    errors_df.index.name = "dt"

    method_name_and_func = {
        "newmark": compute_newmark_error,
        "alpha": compute_alpha_error,
        "erk4": compute_erk4_error,
        "sie": compute_sie_error,
        "mid": compute_mid_error,
    }
    coupling_scheme = "implicit-cps"
    for method_name, method_func in method_name_and_func.items():
        errors_df["error"] = np.array(
            [
                max_norm(
                    method_func(
                        t_stop,
                        N,
                        coupling_scheme,
                        interpolation_order=1,
                    )
                )
                for N in N_list
            ]
        )
        errors_df.to_csv(f"partitioned_{method_name}_waveform.csv")
