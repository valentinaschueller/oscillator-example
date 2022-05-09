import numpy as np
import pandas as pd
import run_partitioned_simulation as rps
from timescales import TimescalesPart
from utility import comment_meta_information


def compute_newmark_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    num_sol = rps.partitioned_newmark_beta(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return num_sol


def compute_alpha_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    num_sol = rps.partitioned_generalized_alpha(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return num_sol


def compute_erk4_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    num_sol = rps.partitioned_erk(
        t_stop, N, 4, TimescalesPart, coupling_scheme, **kwargs
    )
    return num_sol


def compute_sie_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    num_sol = rps.partitioned_semi_implicit_euler(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return num_sol


def compute_mid_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    num_sol = rps.partitioned_implicit_midpoint(
        t_stop, N, TimescalesPart, coupling_scheme, **kwargs
    )
    return num_sol


if __name__ == "__main__":
    """
    Runs partitioned experiment for oscillator example with different time stepping schemes using implicit coupling scheme (fixed-point CPS).
    Outputs trajectory of example.
    """
    t_stop = 5
    N = 1000
    dt = t_stop/N
    timesteps = np.arange(0, t_stop+dt, dt)
    sampling_frequency = 1
    timesteps = timesteps[::sampling_frequency]



    # prepare dataframe for saving
    trajectory_df = pd.DataFrame(index=timesteps)
    trajectory_df.index.name = "t"

    method_name_and_func = {
        "alpha": compute_alpha_trajectory,
        "sie": compute_sie_trajectory,
        "mid": compute_mid_trajectory,
    }
    coupling_scheme = "strang"
    for method_name, method_func in method_name_and_func.items():
        solution = method_func(t_stop, N, coupling_scheme)
        trajectory_df["u1"] = solution[0,::sampling_frequency]
        trajectory_df["u2"] = solution[1,::sampling_frequency]
        trajectory_df["v1"] = solution[2,::sampling_frequency]
        trajectory_df["v2"] = solution[3,::sampling_frequency]
        trajectory_df.to_csv(f"energy_{method_name}_{coupling_scheme}.csv")
        comment_meta_information(method_name+'_'+coupling_scheme, __file__, f"energy_{method_name}_{coupling_scheme}.csv")
