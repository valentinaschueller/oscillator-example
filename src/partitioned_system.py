import numpy as np

from utility import *
from timestepping import ERK, GeneralizedAlpha, NewmarkBeta

def analytical_solution(t_end: float, N: int):
    t = np.linspace(0, t_end, N+1)
    result = np.array([
        0.5 * (np.cos(t) + np.cos(np.sqrt(3)*t)),
        0.5 * (np.cos(t) - np.cos(np.sqrt(3)*t)),
        0.5 * (- np.sin(t) - np.sqrt(3) * np.sin(np.sqrt(3)*t)),
        0.5 * (- np.sin(t) + np.sqrt(3) * np.sin(np.sqrt(3)*t)),
    ])
    return result

def run_cps_simulation(left_system, left_solver, right_system, right_solver, t_end, N):
    dt = t_end / N
    t = 0
    iter = 0
    while iter < N:
        left_system.result[:, iter + 1] = np.squeeze(left_solver.compute_timestep(dt, t, left_system.result[:, iter]))
        right_system.result[:, iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, iter]))
        iter += 1
        t += dt
        left_system.other_u = right_system.result[0, iter]
        right_system.other_u = left_system.result[0, iter]
    return left_system.result, right_system.result

def run_css_simulation(left_system, left_solver, right_system, right_solver, t_end, N):
    dt = t_end / N
    t = 0
    iter = 0
    while iter < N:
        left_system.result[:, iter + 1] = np.squeeze(left_solver.compute_timestep(dt, t, left_system.result[:, iter]))
        right_system.other_u = left_system.result[0, iter + 1]
        right_system.result[:, iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, iter]))
        iter += 1
        t += dt
        left_system.other_u = right_system.result[0, iter]
    return left_system.result, right_system.result

def partitioned_newmark_beta(t_end: float, N: int, serial: bool = False):
    gamma = 0.5
    beta = 0.25
    left_system = SystemPartition(left_system=True, result_shape=(3, N+1))
    right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    # create solvers
    solver_left = NewmarkBeta(
        left_system.A_second_order, left_system.M, left_system.K,
        beta, gamma, left_system.second_order_force
    )
    solver_right = NewmarkBeta(
        right_system.A_second_order, right_system.M, right_system.K,
        beta, gamma, right_system.second_order_force
    )
    if serial:
        left_result, right_result = run_css_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    else:
        left_result, right_result = run_cps_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def partitioned_generalized_alpha(t_end: float, N: int, serial: bool = False):
    alpha_m = 0.3
    alpha_f = 0.5
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (gamma + 0.5)**2

    left_system = SystemPartition(left_system=True, result_shape=(3, N+1))
    right_system = SystemPartition(left_system=False, result_shape=(3, N+1))
    # create solvers
    solver_left = GeneralizedAlpha(
        left_system.A_second_order, left_system.M, left_system.K,
        beta, gamma, alpha_f, alpha_m, left_system.second_order_force
    )
    solver_right = GeneralizedAlpha(
        right_system.A_second_order, right_system.M, right_system.K,
        beta, gamma, alpha_f, alpha_m, right_system.second_order_force
    )
    if serial:
        left_result, right_result = run_css_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    else:
        left_result, right_result = run_cps_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def partitioned_erk(t_end: float, N: int, order: int = 1, serial: bool = False):
    left_system = SystemPartition(left_system=True, result_shape=(2, N+1))
    right_system = SystemPartition(left_system=False, result_shape=(2, N+1))
    # create solvers
    solver_left = ERK(left_system.A_first_order, left_system.first_order_force, order)
    solver_right = ERK(right_system.A_first_order, right_system.first_order_force, order)
    if serial:
        left_result, right_result = run_css_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    else:
        left_result, right_result = run_cps_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result


class SystemPartition:
    def __init__(self, left_system: bool, result_shape: tuple = (3,1)):
        self.left_system_bool = left_system
        # for the formulation: u'' = A_second_order * u:
        self.A_second_order = (-2) * np.eye(1, 1, dtype=float)
        # for the formulation: Mu'' + Ku = 0:
        self.M = np.eye(1, 1, dtype=float)
        self.K = - self.A_second_order.copy()
        # for the formulation: y' = A_first_order * y:
        self.A_first_order = np.array(
            [[0., 1.],
            [-2., 0.]]
        )
        self.result = np.zeros(result_shape)
        self.other_u = self._initial_other_u()
        try:
            self.result[:, 0] = self._initial_conditions()
        except ValueError:
            self.result[:, 0] = self._initial_conditions()[:2]

    def second_order_force(self, t):
        del t # no time-dependent force so far
        # k12 * other_u
        return self.other_u

    def first_order_force(self, t):
        del t # no time-dependent force so far
        # [0, k12 * other_u]
        return np.array([0., self.other_u], dtype=object)

    def _initial_conditions(self):
        if self.left_system_bool:
            u0 = np.array([1.])
        else:
            u0 = np.array([0.])
        v0 = np.array([0.])
        a0 = np.dot(self.A_second_order, u0) + self.second_order_force(0)
        return np.concatenate([u0, v0, a0])
    
    def _initial_other_u(self):
        if not self.left_system_bool:
           return np.array([1.])
        else:
            return np.array([0.])

def compute_newmark_error(t_stop, N):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_newmark_beta(t_stop, N, 0)
    return true_sol - num_sol

def compute_alpha_error(t_stop, N):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_generalized_alpha(t_stop, N, 0)
    return true_sol - num_sol

def compute_erk_error(t_stop, N):
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_erk(t_stop, N, 4, 0)
    return true_sol - num_sol

if __name__ == '__main__':
    N = 100
    t_stop = 20
    true_sol = analytical_solution(t_stop, N)
    num_sol = partitioned_generalized_alpha(t_stop, N, 1)
    t = np.linspace(0., t_stop, N+1)
    create_solution_plots(t, num_sol, f"partitioned_alpha_{N}")
    create_solution_plots(t, true_sol, "analytic")

    title = "Partitioned System: Convergence Plot"
    subtitle = r"$\alpha_m = 0.5, \alpha_f = 0.5$"
    xlabel = "dt"
    ylabel = r"$\vert\vert e \vert\vert_2$"
    fig, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    plot_error_ref(ax, dt_list)
    ax.plot(dt_list, l2_errors_newmark, linestyle="none", marker=".", color="C9", label=r"Newmark $\beta$")
    ax.plot(dt_list, l2_errors_alpha, linestyle="none", marker="x", color="C6", label=r"Generalized $\alpha$")
    ax.plot(dt_list, l2_errors_erk, linestyle="none", marker="1", color="C8", label=r"ERK4")
    ax.legend()
    plt.savefig("convergence_partitioned_all.png", dpi=300, bbox_inches='tight')