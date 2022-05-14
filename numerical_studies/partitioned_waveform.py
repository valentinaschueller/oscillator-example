import numpy as np
import pandas as pd

import compute_partitioned_errors as cpe
from utility import comment_meta_information, max_norm

if __name__ == "__main__":
    """
    Runs partitioned experiment for oscillator example with different time stepping schemes using waveform iterations.
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

    method_name_and_func = {
        "newmark": cpe.compute_newmark_error,
        "alpha": cpe.compute_alpha_error,
        "erk4": cpe.compute_erk4_error,
        "sie": cpe.compute_sie_error,
        "mid": cpe.compute_mid_error,
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
        comment_meta_information(
            method_name + "_waveform",
            __file__,
            f"partitioned_{method_name}_waveform.csv",
        )
