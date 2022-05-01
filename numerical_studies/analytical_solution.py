import timescales
import numpy as np
import pandas as pd

t_end = 20
N = 1000

t = np.linspace(0, t_end, N + 1)
ana_sol_df = pd.DataFrame(index=t)
ana_sol_df.index.name = "t"

st_sol = timescales.analytical_solution(t_end, N)
ana_sol_df["u1"] = st_sol[0]
ana_sol_df["u2"] = st_sol[1]
ana_sol_df["v1"] = st_sol[2]
ana_sol_df["v2"] = st_sol[3]
ana_sol_df.to_csv("analytic.csv")
