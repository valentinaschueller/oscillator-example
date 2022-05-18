import compute_trajectories as ct
import numpy as np
import pandas as pd
from utility import comment_meta_information

if __name__ == "__main__":
    """
    Runs partitioned experiment for oscillator example with different time stepping schemes using implicit coupling scheme (fixed-point CPS).
    Outputs trajectory of example.
    """
    t_stop = 5
    N = 1000
    dt = t_stop / N
    timesteps = np.arange(0, t_stop + dt, dt)
    sampling_frequency = 1
    timesteps = timesteps[::sampling_frequency]

    # prepare dataframe for saving
    trajectory_df = pd.DataFrame(index=timesteps)
    trajectory_df.index.name = "t"

    method_name_and_func = {
        "alpha": ct.compute_alpha_trajectory,
        "sie": ct.compute_sie_trajectory,
        "mid": ct.compute_mid_trajectory,
    }
    coupling_scheme = "implicit-cps"
    for method_name, method_func in method_name_and_func.items():
        solution = method_func(t_stop, N, coupling_scheme, interpolation_order=1)
        trajectory_df["u1"] = solution[0, ::sampling_frequency]
        trajectory_df["u2"] = solution[1, ::sampling_frequency]
        trajectory_df["v1"] = solution[2, ::sampling_frequency]
        trajectory_df["v2"] = solution[3, ::sampling_frequency]
        trajectory_df.to_csv(f"energy_{method_name}_waveform.csv")
        comment_meta_information(
            method_name + "_waveform", __file__, f"energy_{method_name}_waveform.csv"
        )
