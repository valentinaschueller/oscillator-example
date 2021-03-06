import compute_partitioned_errors as cpe
import numpy as np
import pandas as pd
from utility import comment_meta_information, l2_norm_normalized, max_norm

if __name__ == "__main__":
    """
    Runs partitioned experiment for oscillator example with different time stepping schemes using Strang splitting.
    Performs convergence study and outputs error w.r.t analytical solution.
    """
    t_stop = 1
    N_list = np.array([25 * 2**i for i in range(9)])
    dt_list = np.array([t_stop / N for N in N_list])

    # prepare dataframe for saving
    errors_df = pd.DataFrame(index=dt_list)
    errors_df.index.name = "dt"
    use_norm = "max_norm"

    if use_norm == "max_norm":
        norm = max_norm
    elif use_norm == "l2_norm":
        norm = l2_norm_normalized

    method_name_and_func = {
        "newmark": cpe.compute_newmark_error,
        "alpha": cpe.compute_alpha_error,
        "erk4": cpe.compute_erk4_error,
        "sie": cpe.compute_sie_error,
        "mid": cpe.compute_mid_error,
    }

    coupling_scheme = "strang"
    for method_name, method_func in method_name_and_func.items():
        errors_df["error"] = np.array(
            [
                norm(
                    method_func(
                        t_stop,
                        N,
                        coupling_scheme,
                    )
                )
                for N in N_list
            ]
        )
        filename = f"results/partitioned_{method_name}_{coupling_scheme}.csv"
        errors_df.to_csv(filename)
        comment_meta_information(method_name + "_" + coupling_scheme, __file__, filename)
