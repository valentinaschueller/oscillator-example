from same_timescales import SameTimescales, create_solution_plots
import timestepping
import numpy as np
import matplotlib.pyplot as plt

st_mono = SameTimescales()
midpoint = timestepping.ImplicitMidpoint(
    st_mono.A_first_order, st_mono.first_order_force
)
newmark = timestepping.NewmarkBeta(
    st_mono.A_second_order, st_mono.M, st_mono.K, 0.25, 0.5, st_mono.second_order_force
)
erk1 = timestepping.ERK(st_mono.A_first_order, st_mono.first_order_force, 1)
erk2 = timestepping.ERK(st_mono.A_first_order, st_mono.first_order_force, 2)
euler_a = timestepping.SemiImplicitEuler(
    st_mono.A_first_order, st_mono.second_order_force
)

t_end = 200
N = 1000
t = np.linspace(0, t_end, N + 1)

fig, axs = plt.subplots(1, 4, figsize=(30, 10))
fig.suptitle("Phase plots for the simple ODE system", fontsize=14)

nsol = st_mono.numerical_solution(t_end, N, midpoint)
create_solution_plots(t, nsol)
# create_solution_plots(t, asol, "analytical")
u1 = nsol[0]
v1 = nsol[2]
axs[0].plot(u1, v1)
axs[0].set_title(rf"Implicit Midpoint Method: N = {N}, $t_E$ = {t_end}")


nsol = st_mono.numerical_solution(t_end, N, erk2)
t = np.linspace(0, t_end, N + 1)
create_solution_plots(t, nsol)
u1 = nsol[0]
v1 = nsol[2]
axs[1].plot(u1, v1)
axs[1].set_title(rf"Heun's Method: N = {N}, $t_E$ = {t_end}")

nsol = st_mono.numerical_solution(t_end, N, euler_a)
create_solution_plots(t, nsol)
u1 = nsol[0]
v1 = nsol[2]
axs[2].plot(u1, v1)
axs[2].set_title(rf"Semi-Implicit Euler Method: N = {N}, $t_E$ = {t_end}")

asol = st_mono.analytical_solution(t_end, N)
u1 = asol[0]
v1 = asol[2]
axs[3].plot(u1, v1)
axs[3].set_title(rf"Analytical solution")


for ax in axs:
    # ax.set_aspect('equal', 'box')
    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$v_1$")

plt.savefig("phase_space.png", dpi=300, bbox_inches="tight")
