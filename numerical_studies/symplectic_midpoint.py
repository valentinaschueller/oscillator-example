import numpy as np
import pandas as pd
from run_partitioned_simulation import partitioned_implicit_midpoint
from same_timescales import SameTimescalesPart, analytical_solution

if __name__ == "__main__":
    t_end = 40
    N = 500
    t = np.linspace(0, t_end, N + 1)

    phase_df = pd.DataFrame(index=t)
    phase_df.index.name = "t"

    asol = analytical_solution(t_end, N)
    phase_df["u1_ref"] = asol[0]
    phase_df["v1_ref"] = asol[2]

    for coupling_scheme in ["cps", "css", "implicit-cps", "strang"]:
        nsol = partitioned_implicit_midpoint(
            t_end, N, SameTimescalesPart, coupling_scheme
        )
        phase_df["u1"] = nsol[0]
        phase_df["v1"] = nsol[2]
        phase_df.to_csv(f"phase_space_implicit_midpoint_{coupling_scheme}.csv")
    nsol = partitioned_implicit_midpoint(
        t_end, N, SameTimescalesPart, "implicit-cps", interpolation_order=1
    )
    phase_df["u1"] = nsol[0]
    phase_df["v1"] = nsol[2]
    phase_df.to_csv("phase_space_implicit_midpoint_waveform.csv")
