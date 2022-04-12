import diff_timescales as dt
import numpy as np
import pandas as pd
import same_timescales as st

t_end = 20
N = 1000

t = np.linspace(0, t_end, N + 1)
ana_sol_df = pd.DataFrame(index=t)
ana_sol_df.index.name = "t"

st_sol = st.analytical_solution(t_end, N)
ana_sol_df["u1"] = st_sol[0]
ana_sol_df["u2"] = st_sol[1]
ana_sol_df["v1"] = st_sol[2]
ana_sol_df["v2"] = st_sol[3]
ana_sol_df.to_csv("analytic_same_timescales.csv")

dt_sol = dt.analytical_solution(t_end, N)
ana_sol_df["u1"] = dt_sol[0]
ana_sol_df["u2"] = dt_sol[1]
ana_sol_df["v1"] = dt_sol[2]
ana_sol_df["v2"] = dt_sol[3]
ana_sol_df.to_csv("analytic_diff_timescales.csv")
